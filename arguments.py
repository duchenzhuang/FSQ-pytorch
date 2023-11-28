"""argparser configuration"""

import argparse
import os
import torch
from util import multiplyList

def add_model_config_args(parser):
    """Model arguments"""
    group = parser.add_argument_group('model', 'model configuration')

    # Model Encoder
    group.add_argument('--in-channel', type=int, default=3,
                        help="Input Channel")
    group.add_argument('--channel', type=int, default=512,
                        help="Middle Channel")
    
    # VQ-VAE Quantizer
    group.add_argument('--quantizer', type=str, default="ema", choices=['ema','origin','fsq','lfq'],
                       help="use which quantizer")
    #FSQ
    group.add_argument('--levels', nargs='+',type=int, default=[8,5,5,5], help='fsq levels')
    #LFQ
    group.add_argument('--lfq-dim', type=int, default=10,
                        help="look up free quantizer dim")
    group.add_argument('--entropy-loss-weight', type=float, default=0.)
    group.add_argument('--codebook-loss-weight', type=float, default=0.)
    
    group.add_argument('--embed-dim', type=int, default=256,
                        help="The embedding dimension of VQVAE's codebook")
    group.add_argument('--n-embed', type=int, default=1024,
                        help="The embedding dimension of VQVAE's codebook")
    return parser

def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--experiment-name', type=str, default="VQVAE",
                       help="The experiment name for summary and checkpoint")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--weight-decay', type=float, default=0.,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=500000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--max-train-epochs', type=int, default=100,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--log-interval', type=int, default=100,
                       help='report interval')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed')

    
    # Learning rate.
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--recon-save', action='store_true', default=True,
                       help='Whether to save reconstruction sample')
    group.add_argument('--save-interval', type=int, default=5000,#5000
                       help='number of iterations between saves')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
   
    
    # distributed training args
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'training. One of [gloo, nccl]')
    group.add_argument('--local-rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    
    # part to train/finetune
    group.add_argument('--train-modules', type=str, default='all',
                       help='Module to train')
    
    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument('--train-data-path', type=str, default="/localdata_ssd/ImageNet_ILSVRC2012/train",
                        help="the path of training data")
    group.add_argument('--val-data-path', type=str, default="/localdata_ssd/ImageNet_ILSVRC2012/val", 
                       help="the path of val data")
    group.add_argument('--img-size', type=int, default=128,
                        help='Size of image for dataset')
    group.add_argument('--num-workers', type=int, default=8,
                       help="""Number of workers to use for dataloading""")
    return parser

def add_loss_args(parser):
    """Loss weight arguments"""
    
    group = parser.add_argument_group('weight', 'loss configurations')
   
    group.add_argument('--l1-weight', type=float, default=1.,
                       help='Weight of l1 loss.')
    group.add_argument('--perceptual-weight', type=float, default=1.,
                       help='Weight of perceptual loss.')
    group.add_argument('--codebook-weight', type=float, default=1.,
                       help='Weight of codebook loss.')
    return parser
 


def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch VQVAE')
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_data_args(parser)
    parser = add_loss_args(parser)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    args.save = 'checkpoints/' + str(args.quantizer) 

    if args.quantizer == 'lfq':
        args.embed_dim = args.lfq_dim
        args.save += '-entropy_weights-' + str(args.entropy_loss_weight)
        args.save += '-codebook_weights-' + str(args.codebook_loss_weight)
        args.save += '-lfq_dim-' + str(args.lfq_dim)
        
    elif args.quantizer == 'fsq':
        args.embed_dim = len(args.levels)
        args.save += '-n_embed-' + str(multiplyList(args.levels))

    else:
        args.save += '-n_embed-' + str(args.n_embed)
    return args
