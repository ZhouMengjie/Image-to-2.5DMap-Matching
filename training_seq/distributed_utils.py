import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        port = args.port
        # os.environ['MASTER_PORT'] = port
        print('[CONFIG] distributed init (local rank {})'.format(args.gpu), flush=True)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = args.rank % torch.cuda.device_count()
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')
        port = args.port
        os.environ['MASTER_PORT'] = port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        print('[CONFIG] distributed init (local rank {})'.format(args.gpu), flush=True)
    else:
        print('Not using distributed mode')
        args.log = True
        args.device = torch.device('cpu')
        return

    args.device = torch.device('cuda', args.rank)  # rank or gpu? gpu == local rank
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl' # nccl is recommended for Nvidia GPUs
    args.dist_url = 'env://'
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    args.log = (args.rank == 0)    
    print("[CONFIG] Let's use", torch.cuda.device_count(), "GPUs!")
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """check whether support the distributed environment"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # single GPU
        return value

    with torch.no_grad():
        dist.all_reduce(value) # value must be a tensor
        if average:
            value /= world_size

        return value
