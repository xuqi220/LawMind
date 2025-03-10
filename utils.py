from transformers import AutoTokenizer
from model.config import LMConfig
from model.model import LawMindLM
import os
import numpy as np
import random
import torch
from torch.distributed import init_process_group
cwd = os.getcwd()

def Logger(content, is_master_process):
    if is_master_process:
        print(content)

def init_model(lm_config, args):
    tokenizer = AutoTokenizer.from_pretrained(f'{cwd}/LawMind/model/LawMind_tokenizer')
    model = LawMindLM(lm_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.dtype=="bfloat16":
        mem = 2*trainable_params # byte
    if args.dtype=="fp32":
        mem = 4*trainable_params # byte
    return model, tokenizer,f'LLM总参数量：{ trainable_params/ 1e6:.3f} millions | Mem: {mem/ 1e6:.3f} MB'


def load_tokens(filename):
    npt = np.memmap(filename, dtype=np.uint16, mode='r')
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# 获取分布式训练配置信息
def get_ddp_info():
    is_ddp = int(os.environ.get("RANK", -1))!=-1
    if is_ddp:# 当前处于ddp训练环境下
        # 初始化进程组
        init_process_group('nccl') 
        rank = int(os.environ.get("RANK"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master_process = rank==0
    else: # 非ddp环境
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cpu"
        if torch.cuda.is_available():  
            device = "cuda:{local_rank}"
            torch.cuda.set_device(device)
        elif torch.backends.mps.is_available():
            device = "mps"
        is_master_process = True
    return rank, local_rank, world_size, device, is_master_process

def set_seed():
    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1024)

class DistDataLoader():
    def __init__(self):
        pass