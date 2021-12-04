import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation
from torchvision.models.mobilenetv3 import (
    SqueezeExcitation,
    InvertedResidualConfig,
    InvertedResidual,
)

from src.modules.base_generator import GeneratorAbstract


class MobileNetV3(nn.Module):
    """refer to mobilenetv3.py at torchvision for the details"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int = 960,
        pretrained: bool = True,
    ):
        # initial settings
        super(MobileNetV3, self).__init__()
        self.model = models.mobilenet.mobilenet_v3_large(pretrained=pretrained)

        # conditional elimination on convolutional layers based on desired output channel
        if out_channel == 960:
            pass
        elif out_channel == 160:
            # exlude the last block from model.features
            self.model.features = self.model.features[:-1]
        elif out_channel == 112:
            # delete last four blocks from model.features
            self.model.features = self.model.features[:-4]
            pass
        elif out_channel == 80:
            # delete last six blocks from model.features
            self.model.features = self.model.features[:-6]
            pass
        else:
            raise Exception("out_channel: 960, 160, 112 or 80")
        # delete the head of the model layers
        del self.model.avgpool
        del self.model.classifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        return x


class MobileNetV3Generator(GeneratorAbstract):
    """MobileNetV3 (torchvision.models) module generator for parsing."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.args[0]

    def __call__(self, repeat: int = 1):
        # TODO: Apply repeat
        pretrained = self.args[1] if len(self.args) > 1 else True

        return self._get_module(
            MobileNetV3(self.in_channel, self.out_channel, pretrained=pretrained)
        )
