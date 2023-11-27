import os
import torch
from arguments import get_args
from model import VQVAE
from dataset import get_data_loaders
from util import initialize_distributed, set_random_seed, mkdir_ckpt_dirs
from scheduler import AnnealingLR
from lpips import LPIPS
from metric import get_revd_perceptual
from torchvision.utils import save_image, make_grid

def main():
    args = get_args()
    print(args)
    initialize_distributed(args)
    set_random_seed(args.seed)
    mkdir_ckpt_dirs(args)

    # 1, load dataset
    train_data_loader, val_data_loader = get_data_loaders(args)

    # 2, load model
    model = VQVAE(args)
    model.cuda(torch.cuda.current_device())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)

    # 3, load optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr = args.lr,
        warmup_iter=args.warmup*args.train_iters,
        num_iters=args.train_iters,
        decay_style=args.lr_decay_style,
        last_iter=-1,
        decay_ratio=args.lr_decay_ratio
    )

    # 4. load perceptual model
    perceptual_model = LPIPS().eval()
    perceptual_model.cuda(torch.cuda.current_device())

    torch.distributed.barrier()
    # 5. begin training
    num_iter = 0
    get_l1loss = torch.nn.L1Loss()

    for epoch in range(args.max_train_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        for _, (input_img,_) in enumerate(train_data_loader):
            num_iter += 1
            # test saving
            if num_iter == 1:
                torch.save({
                    'iter': num_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
                }, args.save+"/ckpts/{}.pt".format(epoch))
            
            # forward
            input_img = input_img.cuda(torch.cuda.current_device())
            reconstructions, codebook_loss = model(input_img)
            l1loss = get_l1loss(input_img, reconstructions)
            perceptual_loss = get_revd_perceptual(input_img, reconstructions,perceptual_model)
            loss = codebook_loss + l1loss + perceptual_loss 
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # print info
            if torch.distributed.get_rank() == 0 and num_iter % 5 == 0:
                print("rank 0: epoch:{}, iter:{}, lr:{:.4}, l1loss:{:.4}, percep_loss:{:.4}, codebook_loss:{:.4}".format(epoch, num_iter,optimizer.state_dict()['param_groups'][0]['lr'] ,l1loss.item(), perceptual_loss.item(), codebook_loss.item()))
            
            # save image for checking training
            if num_iter % args.log_interval == 0 and torch.distributed.get_rank() == 0:
                save_image(make_grid(torch.cat([input_img, reconstructions]), nrow=input_img.shape[0]), args.save+'/samples/{}.jpg'.format(num_iter), normalize=True)
        
        # save checkpoints
        if epoch % 5 == 0 and torch.distributed.get_rank() == 0:
            torch.save({
                'iter': num_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }, args.save+"/ckpts/{}.pt".format(epoch))






if __name__ == "__main__":
    main()