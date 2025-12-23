import tomllib
import torch
import torch.nn as nn
from tmp.model.registry import get_module


class MicroDet(nn.Module):
    def __init__(self, cfg_path: str):
        super().__init__()
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)

        self.cfg = cfg["model"]
        self.nc = self.cfg.get("nc", 1)
        self.names = self.cfg.get("names", ["object"])

        self.layers, self.save, self.out_channels = self._build_model(self.cfg)

        print(f"MicroDet built with {self.nc} classes.")

    def _build_model(self, cfg):
        layers = nn.ModuleList()
        save = []
        ch = []  
       
        ch_in = 3

        for i, block in enumerate(cfg["backbone"] + cfg["neck"] + cfg["head"] + cfg["detect"]):
            mname = block["module"]
            module = get_module(mname)

            from_idx = block.get("from", -1)
            repeats = block.get("repeats", 1)
            args = block.get("args", [])

            
            if isinstance(from_idx, int):
                ch_in = ch[from_idx] if from_idx != -1 else ch_in
            else:  
                ch_in = sum(ch[j] for j in from_idx)

            
            if "in_channels" in module.__init__.__code__.co_varnames:
                args = [ch_in] + args

          
            seq = nn.Sequential(*[module(*args) for _ in range(repeats)])
            layers.append(seq)

            
            if hasattr(seq[-1], "out_channels"):
                ch_out = seq[-1].out_channels
            else:
                ch_out = ch_in

            ch.append(ch_out)

            if isinstance(from_idx, list):
                save.extend(from_idx)

        return layers, save, ch

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.layers):
            if self.cfg["backbone"][i].get("from", -1) != -1:
                idx = self.cfg["backbone"][i]["from"]
                if isinstance(idx, int):
                    x = module(outputs[idx])
                else:
                    x = module(torch.cat([outputs[j] for j in idx], 1))
            else:
                x = module(x)

            outputs.append(x)

        return outputs[-1]
