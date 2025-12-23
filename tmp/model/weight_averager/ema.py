from tmp.src.common_imports  import torch, nn, copy


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, device=None):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
                else:
                    v.copy_(msd[k])

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "decay": self.decay}

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"])
        self.decay = state.get("decay", self.decay)
