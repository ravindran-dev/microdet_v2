from tmp.src.common_imports import torch, nn


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0, requires_grad: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
