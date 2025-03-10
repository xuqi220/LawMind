import numpy as np
import tiktoken
import torch


a = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)

b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)

print(a*b)

