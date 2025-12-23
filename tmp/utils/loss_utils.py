import torch
import torch.nn as nn
import torch.nn.functional as F


class Integral(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x):
        s = x.size()
        prob = F.softmax(x.view(*s[:-1], 4, self.reg_max + 1), dim=-1)
        return F.linear(prob, self.project.to(prob)).view(*s[:-1], 4)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    out = tensor.clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
    out /= torch.distributed.get_world_size()
    return out
