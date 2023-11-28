import torch
import os
import random
import numpy as np

def initialize_distributed(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def mkdir_ckpt_dirs(args):
    if torch.distributed.get_rank() == 0:
        if os.path.exists(args.save):
            print('savedir already here.', args.save)
            exit()
        else:
            os.makedirs(args.save)
            os.makedirs(args.save + '/ckpts')
            os.makedirs(args.save + '/samples')
        
        argsDict = args.__dict__
        with open(os.path.join(args.save, 'setting.txt'), 'w') as f:
            f.writelines('------------------- start -------------------' + '\n')
            for arg, value in argsDict.items():
                f.writelines(arg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------' + '\n')

def multiplyList(myList):
 
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result