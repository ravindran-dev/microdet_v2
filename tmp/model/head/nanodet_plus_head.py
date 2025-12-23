import torch
import torch.nn as nn
from typing import List, Sequence, Any
from tmp.model.module.conv import ConvModule, DepthwiseConvModule
from tmp.model.registry import register_module


@register_module
class NanoDetPlusHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        loss: Any,
        input_channel: int,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        kernel_size: int = 5,
        strides: Sequence[int] = (8, 16, 32),
        conv_type: str = "DWConv",
        norm_cfg: Any = None,
        reg_max: int = 7,
        activation: str = "LeakyReLU",
        assigner_cfg: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(input_channel)
        self.feat_channels = int(feat_channels)
        self.stacked_convs = int(stacked_convs)
        self.kernel_size = int(kernel_size)
        self.strides = list(strides)
        self.reg_max = int(reg_max)
        self.activation = activation
        self.Conv = DepthwiseConvModule if conv_type == "DWConv" else ConvModule

        self.stems = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for _ in self.strides:
            stem_layers = []
            for i in range(self.stacked_convs):
                in_ch = self.in_channels if i == 0 else self.feat_channels
                stem_layers.append(
                    self.Conv(
                        in_ch,
                        self.feat_channels,
                        self.kernel_size,
                        1,
                        self.kernel_size // 2,
                        activation=self.activation,
                    )
                )
            self.stems.append(nn.Sequential(*stem_layers))
            self.cls_heads.append(
                nn.Conv2d(self.feat_channels, self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_heads.append(
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), kernel_size=1, stride=1, padding=0)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, feats: Sequence[torch.Tensor]) -> List[dict]:
        outs: List[dict] = []
        for feat, stem, cls_head, reg_head in zip(feats, self.stems, self.cls_heads, self.reg_heads):
            x = stem(feat)
            cls_logits = cls_head(x)
            reg_logits = reg_head(x)
            outs.append({"cls": cls_logits, "reg": reg_logits})
        return outs
