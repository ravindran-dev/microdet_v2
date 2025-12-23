import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from tmp.model.registry import register_module

def act_layers(name: str):
    name = (name or "ReLU").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("leakyrelu", "leaky"):
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)

MODEL_URLS = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
}

def channel_shuffle(x: torch.Tensor, groups: int = 2):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class DWConvBNAct(nn.Module):
    def __init__(self, c, k=3, s=1, p=1, act=True, act_name="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, s, p, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = act_layers(act_name) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PWConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, act=True, act_name="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = act_layers(act_name) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ShuffleV2Unit(nn.Module):
    def __init__(self, c_in, c_out, stride, act_name="ReLU"):
        super().__init__()
        self.stride = stride
        branch_out = c_out // 2
        if stride == 1:
            self.branch1 = None
            in_branch2 = c_in // 2
        else:
            self.branch1 = nn.Sequential(
                DWConvBNAct(c_in, 3, 2, 1, act=False),
                PWConvBNAct(c_in, branch_out, act=True, act_name=act_name),
            )
            in_branch2 = c_in

        self.branch2 = nn.Sequential(
            PWConvBNAct(in_branch2, branch_out, act=True, act_name=act_name),
            DWConvBNAct(branch_out, 3, stride, 1, act=False),
            PWConvBNAct(branch_out, branch_out, act=True, act_name=act_name),
        )

    def forward(self, x):
        if self.stride == 1:
            c = x.size(1)
            x1, x2 = x[:, : c // 2], x[:, c // 2 :]
            out = torch.cat([x1, self.branch2(x2)], 1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], 1)
        return channel_shuffle(out, 2)

@register_module
class ShuffleNetV2(nn.Module):
    def __init__(self, model_size="1.0x", out_stages=(2, 3, 4), activation="LeakyReLU", pretrain=True):
        super().__init__()
        self.model_size = model_size
        self.out_stages = tuple(sorted(out_stages))
        self.activation = activation

        if model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
            repeats = (4, 8, 4)
        elif model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
            repeats = (4, 8, 4)
        else:
            raise ValueError("model_size must be '0.5x' or '1.0x'")

        stem_c, s8_c, s16_c, s32_c, _ = self._stage_out_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_c),
            act_layers(activation),
            nn.MaxPool2d(3, 2, 1),
        )

        self.stage2 = self._make_stage(stem_c, s8_c, repeats[0], 2)
        self.stage3 = self._make_stage(s8_c, s16_c, repeats[1], 2)
        self.stage4 = self._make_stage(s16_c, s32_c, repeats[2], 2)

        self.channels = {"s8": s8_c, "s16": s16_c, "s32": s32_c}

        self._init_weights()
        if pretrain:
            self._load_imagenet()

    def _make_stage(self, c_in, c_out, repeat, first_stride):
        layers = [ShuffleV2Unit(c_in, c_out, first_stride, self.activation)]
        for _ in range(repeat - 1):
            layers.append(ShuffleV2Unit(c_out, c_out, 1, self.activation))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_imagenet(self):
        key = f"shufflenetv2_{self.model_size}"
        url = MODEL_URLS.get(key)
        if url:
            try:
                state = model_zoo.load_url(url, progress=True)
                self.load_state_dict(state, strict=False)
            except Exception:
                pass

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c2, c3, c4
