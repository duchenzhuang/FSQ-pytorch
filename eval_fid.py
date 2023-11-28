import torch
from arguments import get_args
from model import VQVAE
from dataset import get_transform
from torchvision import datasets, transforms

from torchmetrics.image.fid import FrechetInceptionDistance

def main():
    args = get_args()

    # 1, load dataset
    imagenet_transform = get_transform(args)
    val_set = datasets.ImageFolder(args.val_data_path,imagenet_transform)
    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    transform_rev = transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1. / 0.229, 1. / 0.224, 1. / 0.225])


    # 2, load model
    model = VQVAE(args)
    model.cuda(torch.cuda.current_device())

    # original saved file with DataParallel
    state_dict = torch.load(args.load)['model_state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    #
    fid = FrechetInceptionDistance(feature=2048, normalize=True)

    for i, (input_img,_) in enumerate(val_data_loader):            
        # forward
        print(i*args.batch_size)

        with torch.no_grad():
            input_img = input_img.cuda(torch.cuda.current_device())
            reconstructions, codebook_loss = model(input_img)
        
        input_img = transform_rev(input_img.contiguous())
        reconstructions = transform_rev(reconstructions.contiguous())

        fid.update(input_img.cpu(), real=True)
        fid.update(reconstructions.cpu(), real=False)
        # print info

    print(fid.compute())


if __name__ == "__main__":
    main()