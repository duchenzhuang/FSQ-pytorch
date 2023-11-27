import torch
from torchvision import datasets, transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_transform(args):
    #Train and Val share the same transform
    imagenet_transform = [
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
    ]
    return transforms.Compose(imagenet_transform)

def get_data_loaders(args):
    """
    get a distributed imagenet train dataloader and a non-distributed imagenet val dataloader
    """
    #'/localdata_ssd/ImageNet_ILSVRC2012/train'
    #'/localdata_ssd/ImageNet_ILSVRC2012/val'

    imagenet_transform = get_transform(args)
    train_set = datasets.ImageFolder(args.train_data_path,imagenet_transform)
    val_set = datasets.ImageFolder(args.val_data_path,imagenet_transform)

    sampler_train = torch.utils.data.DistributedSampler(
            train_set, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=True
        )
    train_data_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_data_loader, val_data_loader
    