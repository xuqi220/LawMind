import argparse
import os
import torch
import utils
from model.config import LMConfig
from model.model import LawMindLM
from transformers import AutoTokenizer
cwd = os.getcwd()
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LawMind Pretraining")
    # 模型参数
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=4096, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 模型训练参数
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tokens_per_step", type=int, default=524288)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=f"{cwd}/dataset")
    # log
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="LawMind-Pretrain")
    parser.add_argument("--out_dir", type=str, default=f"{cwd}/out")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    
    # 分布式配置信息 
    rank, local_rank, world_size, device, is_master_process = utils.get_ddp_info()
    # print(f"rank:{rank} | local_rank:{local_rank} | world_size:{world_size} | use:{device} | master_process: {is_master_process}")
    
    # 设置随机种子
    utils.set_seed()
    
    # 超参数
    assert args.tokens_per_step%(args.batch_size*args.max_seq_len*world_size)==0 
    args.accumulation_steps = args.tokens_per_step//(args.batch_size*args.max_seq_len*world_size)
    msg = f"accumulation_steps_per_device:{args.accumulation_steps}"
    utils.Logger(msg, is_master_process)
    
    
    
    # 初始化模型配置文件
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    
    
    
    model, tokenizer, msg = utils.init_model(lm_config, args)
    model.to(device)
    utils.Logger(msg, is_master_process)
