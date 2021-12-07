import torch
from torch import nn, Tensor

from src.modules.base_generator import GeneratorAbstract

from torchvision import models


class VGG11(nn.Module):
    """refer to vgg.py at torchvision for the details"""

    def __init__(self, in_channel: int, out_channel: int = 512, pretrained: bool = True):
        # initial settings
        super(VGG11, self).__init__()
        self.model = models.vgg11(pretrained=pretrained)
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        # conditional elimination on convolutional layers based on desired output channel
        if out_channel == 512:
            pass
        elif out_channel == 256:
            pass
        elif out_channel == 128:
            pass

        del self.model.avgpool
        del self.model.classifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class VGG11Generator(GeneratorAbstract):
    """VGG11 (torchvision.models) module generator for parsing."""

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

        return self._get_module(VGG11(self.in_channel, self.out_channel, pretrained=pretrained))
