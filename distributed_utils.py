import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """初始化分布式训练环境"""
    # 获取环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    if local_rank != -1:
        if not dist.is_initialized():
            # 初始化进程组
            dist.init_process_group(backend='nccl')
            # 设置当前设备
            torch.cuda.set_device(local_rank)
            
        print(f"🖥️ 进程 {local_rank}/{world_size-1} 初始化完成")
        return True, local_rank, world_size
    return False, 0, 1

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 分布式训练环境已清理")

def to_distributed(model, local_rank):
    """将模型转换为分布式模型"""
    if dist.is_initialized():
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
        return model
    return model.cuda()

def create_distributed_loader(dataset, batch_size, num_workers=4):
    """创建分布式数据加载器"""
    if dist.is_initialized():
        sampler = DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader, sampler
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader, None

def is_main_process():
    """判断是否为主进程"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()

def reduce_value(value, average=True):
    """归约值到所有进程"""
    if not dist.is_initialized():
        return value
    
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= dist.get_world_size()
    return value