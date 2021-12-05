import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision import models

from src.modules.base_generator import GeneratorAbstract


class EfficientNetB0(nn.Module):
    """refer to efficientnet.py at torchvision for the details"""

    def __init__(
        self,
        in_channels: int,
        out_channel: int = 320,
        pretrained: bool = True,
    ):
        # initial settings
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet.efficientnet_b0(pretrained=pretrained)

        # conditional elimination on convolutional layers based on desired output channel
        if out_channel == 1280:
            pass
        elif out_channel == 320:
            # remove the last convolutional layer
            self.model.features = self.model.features[:-1]
        elif out_channel == 192:
            # remove the last convolutional layer and last mbconv block
            self.model.features = self.model.features[:-2]
        elif out_channel == 112:
            # remove the last convolutional layer and last two mbconv block
            self.model.features = self.model.features[:-3]
        elif out_channel == 80:
            # remove the last convolutional layer and last three mbconv block
            self.model.features = self.model.features[:-4]
        else:
            raise Exception("out_channel should be in [1280, 320, 192, 112, 80]")

        # delete the head of the model layers
        del self.model.avgpool
        del self.model.classifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.features(x)
        return x


class EfficientNetB0Generator(GeneratorAbstract):
    """EfficientNetB0 (torchvision.models) module generator for parsing."""

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
            EfficientNetB0(self.in_channel, self.out_channel, pretrained=pretrained)
        )
