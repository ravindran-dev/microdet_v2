import tomllib
import pathlib
from typing import Any, Dict, List, Sequence, Tuple, Union
import torch
import torch.nn as nn
from tmp.model.registry import get_module


def _safe_instantiate(module_class, args):
    if args is None:
        return module_class()
    if isinstance(args, dict):
        return module_class(**args)
    if isinstance(args, (list, tuple)):
        return module_class(*args)
    return module_class(args)


class ModularLayer(nn.Module):
    def __init__(self, module_name: str, args: Any = None, repeats: int = 1):
        super().__init__()
        module_class = get_module(module_name)
        modules = []
        for _ in range(int(repeats)):
            inst = _safe_instantiate(module_class, args)
            modules.append(inst)
        self.module = modules[0] if len(modules) == 1 else nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class MicroDet(nn.Module):
    def __init__(self, cfg: Union[str, bytes, dict, pathlib.Path]):
        super().__init__()

        # ---------------- config loading ----------------
        if isinstance(cfg, (str, bytes, pathlib.Path)):
            p = pathlib.Path(cfg)
            with open(p, "rb") as f:
                full = tomllib.load(f)
            mcfg = full.get("model", full)
        elif isinstance(cfg, dict):
            mcfg = cfg
        else:
            raise TypeError("cfg must be path or dict")

        self.cfg = mcfg
        self.nc = int(mcfg.get("nc", 1))
        self.names = mcfg.get("names", [f"cls{i}" for i in range(self.nc)])
        self.input_size = tuple(mcfg.get("input_size", [640, 640]))

        # ---------------- global dict ----------------
        head_cfg = mcfg.get("head", {})
        if isinstance(head_cfg, dict):
            loss_cfg = head_cfg.get("loss", {})
            assigner_cfg = head_cfg.get("assigner_cfg", {})
        else:
            loss_cfg = {}
            assigner_cfg = {}

        self.global_dict = {
            "nc": self.nc,
            "loss": loss_cfg,
            "assigner_cfg": assigner_cfg,
        }

        # ---------------- build model ----------------
        self.layers, self.save = self._build_network_and_validate()

        # ---------------- expose strides (IMPORTANT) ----------------
        self.strides = None
        for m in self.modules():
            if hasattr(m, "strides") and m.strides is not None:
                self.strides = list(m.strides)
                break

        if self.strides is None:
            raise RuntimeError(
                "MicroDet could not infer strides from model. "
                "Ensure Detect or Head module defines `self.strides`."
            )

    def _build_network_and_validate(self) -> Tuple[nn.ModuleList, List[Any]]:
        sections = ["backbone", "neck", "head", "detect"]
        items_flat: List[Tuple[str, Any]] = []
        for sec in sections:
            items = self.cfg.get(sec, [])
            if isinstance(items, dict):
                items = [items]
            for it in items:
                cfg_item = it.get("config") if isinstance(it, dict) and "config" in it else it
                items_flat.append((sec, cfg_item))

        all_layers = nn.ModuleList()
        save: List[Any] = []

        for i, (sec, cfg_item) in enumerate(items_flat):
            if not isinstance(cfg_item, (list, tuple)):
                raise ValueError(f"Bad layer config at section '{sec}' index {i}: expected list/tuple, got {type(cfg_item)}. cfg_item={cfg_item}")
            if len(cfg_item) < 3:
                raise ValueError(f"Bad layer config at section '{sec}' index {i}: expected [from, repeats, module_name, args?]. cfg_item={cfg_item}")

            from_idx = cfg_item[0]
            repeats = int(cfg_item[1])
            module_name = cfg_item[2]
            args = cfg_item[3] if len(cfg_item) > 3 else None
            resolved_args = self._resolve_args(args)

            # Validation for `from_idx` (accept int or list/tuple)
            def _check_idx(idx, produced_outputs):
                if idx == -1:
                    return True
                if not isinstance(idx, int):
                    return False
                if idx < 0:
                    return False
                if idx >= produced_outputs:
                    return False
                return True

            produced_outputs = len(all_layers)  # outputs we will have available when this layer runs
            if isinstance(from_idx, (list, tuple)):
                for idx in from_idx:
                    if not _check_idx(idx, produced_outputs):
                        raise ValueError(
                            f"Invalid 'from' index for layer #{i} (section={sec}, module={module_name}). "
                            f"'from' contains {idx} but at build time only {produced_outputs} outputs exist. "
                            f"cfg_item={cfg_item}"
                        )
            else:
                if not _check_idx(from_idx, produced_outputs):
                    raise ValueError(
                        f"Invalid 'from' index for layer #{i} (section={sec}, module={module_name}). "
                        f"'from' is {from_idx} but at build time only {produced_outputs} outputs exist. "
                        f"cfg_item={cfg_item}"
                    )

            layer = ModularLayer(module_name, resolved_args, repeats=repeats)
            all_layers.append(layer)
            save.append(from_idx)

        # print a short summary to help debug configs
        for idx, (layer, f) in enumerate(zip(all_layers, save)):
            mod_name = layer.__class__.__name__ if not isinstance(layer, nn.Sequential) else "Sequential/" + layer[0].__class__.__name__
            print(f"[model_builder] layer #{idx:<3} module={mod_name:<30} from={f}")

        return all_layers, save

    def _resolve_args(self, args: Any) -> Any:
        if isinstance(args, str) and args in self.global_dict:
            return self.global_dict[args]
        if isinstance(args, (list, tuple)):
            out = []
            for a in args:
                if isinstance(a, str) and a in self.global_dict:
                    out.append(self.global_dict[a])
                elif isinstance(a, (list, tuple)):
                    out.append(self._resolve_args(list(a)))
                elif isinstance(a, dict):
                    out.append({k: (self.global_dict[v] if isinstance(v, str) and v in self.global_dict else v) for k, v in a.items()})
                else:
                    out.append(a)
            return out
        if isinstance(args, dict):
            return {k: (self.global_dict[v] if isinstance(v, str) and v in self.global_dict else v) for k, v in args.items()}
        return args

    def forward(self, x):
        outputs: List[Any] = []
        for layer_idx, (layer, f) in enumerate(zip(self.layers, self.save)):
            if isinstance(f, (list, tuple)):
                inp_list = []
                for idx in f:
                    if idx == -1:
                        inp_list.append(x)
                    else:
                        if idx < 0 or idx >= len(outputs):
                            raise IndexError(
                                f"Layer {layer_idx}: 'from' index {idx} is invalid at forward-time (have {len(outputs)} previous outputs)."
                            )
                        inp_list.append(outputs[idx])
                inp = inp_list[0] if len(inp_list) == 1 else inp_list
                out = layer(inp)
            else:
                idx = f
                if idx == -1:
                    inp = x
                else:
                    if idx < 0 or idx >= len(outputs):
                        raise IndexError(
                            f"Layer {layer_idx}: 'from' index {idx} is invalid at forward-time (have {len(outputs)} previous outputs)."
                        )
                    inp = outputs[idx]
                out = layer(inp)
            outputs.append(out)
        if len(outputs) == 0:
            return x
        return outputs[-1]
