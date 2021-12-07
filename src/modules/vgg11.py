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
            self.model.features = self.model.features[:16]
            # Sequential(
            #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (1): ReLU(inplace=True)
            #     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #     (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (4): ReLU(inplace=True)
            #     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #     (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (7): ReLU(inplace=True)
            #     (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (9): ReLU(inplace=True)
            #     (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #     (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (12): ReLU(inplace=True)
            #     (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #     (14): ReLU(inplace=True)
            #     (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            # )
            pass
        elif out_channel == 256:
            self.model.features = self.model.features[:11]
            # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (1): ReLU(inplace=True)
            # (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            # (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (4): ReLU(inplace=True)
            # (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            # (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (7): ReLU(inplace=True)
            # (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (9): ReLU(inplace=True)
            # (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

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
