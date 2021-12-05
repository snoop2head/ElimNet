"""PyTorch Module and ModuleGenerator."""

# common modules and generators
from src.modules.linear import Linear, LinearGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.convresidual import ConvResidual, ConvResidualGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.poolings import (
    AvgPoolGenerator,
    GlobalAvgPool,
    GlobalAvgPoolGenerator,
    MaxPoolGenerator,
)

# MobileNet V2 related modules and generators
from src.modules.invertedresidualv2 import InvertedResidualv2, InvertedResidualv2Generator

# Mobilenet V3 related modules and generators
from src.modules.mobilenetv3 import MobileNetV3, MobileNetV3Generator
from src.modules.invertedresidualv3 import InvertedResidualv3, InvertedResidualv3Generator

# Resnet18 related modules
from src.modules.resnet18 import Resnet18, Resnet18Generator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator

# efficientnet related modules
from src.modules.mbconv import (
    MBConv,
    MBConvGenerator,
    ConvBNReLU,
    SwishImplementation,
    Swish,
    SqueezeExcitation,
)
from src.modules.efficientnetb0 import EfficientNetB0, EfficientNetB0Generator


__all__ = [
    # common modules
    "Linear",
    "Conv",
    "DWConv",
    "ConvResidual",
    "GlobalAvgPool",
    # common generators
    "GeneratorAbstract",
    "LinearGenerator",
    "ConvGenerator",
    "FixedConvGenerator",
    "DWConvGenerator",
    "ConvResidualGenerator",
    "GlobalAvgPoolGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "ModuleGenerator",
    # MobileNet V2 related modules and generators
    "InvertedResidualv2",
    "InvertedResidualv2Generator",
    # Mobilenet V3 related modules and generators
    "MobileNetV3",
    "MobileNetV3Generator",
    "InvertedResidualv3",
    "InvertedResidualv3Generator",
    # Resnet18 related modules
    "Resnet18",
    "Resnet18Generator",
    "BottleneckGenerator",
    "Bottleneck",
    # EfficientNet Related Modules
    "MBConv",
    "ConvBNReLU",
    "SwishImplementation",
    "Swish",
    "SqueezeExcitation",
    "MBConvGenerator",
    "EfficientNetB0",
    "EfficientNetB0Generator",
]
