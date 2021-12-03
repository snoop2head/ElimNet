import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract

from torchvision import models


class Resnet18(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, pretrained: bool):
        """

        Args:
            in_channel: input channels.
            out_channel: output channels.
        """
        super().__init__()
        self.out_channel = out_channel     
        self.model = models.resnet18(pretrained=pretrained)
        del self.model.fc
        del self.model.avgpool
        
        if self.out_channel == 512:
            pass
        elif self.out_channel == 256:
            del self.model.layer4
        elif self.out_channel == 128:
            del self.model.layer4
            del self.model.layer3
        elif self.out_channel == 64:
            del self.model.layer4
            del self.model.layer3
            del self.model.layer2
        else:
            raise Exception("out_channel: 512, 256, 128 or 64")

        
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        if self.out_channel >= 128:
            x = self.model.layer2(x)
        if self.out_channel >= 256:
            x = self.model.layer3(x)
        if self.out_channel >= 512:
            x = self.model.layer4(x)

        return x


class Resnet18Generator(GeneratorAbstract):
    """ Resnet18 (torchvision.models) module generator for parsing."""

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
            Resnet18(self.in_channel, self.out_channel, pretrained=pretrained)
        )