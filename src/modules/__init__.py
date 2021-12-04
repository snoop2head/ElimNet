"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.convresidual import ConvResidual, ConvResidualGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator
from src.modules.invertedresidualv2 import InvertedResidualv2, InvertedResidualv2Generator
from src.modules.invertedresidualv3 import InvertedResidualv3, InvertedResidualv3Generator
from src.modules.linear import Linear, LinearGenerator
from src.modules.resnet18 import Resnet18, Resnet18Generator
from src.modules.poolings import (
    AvgPoolGenerator,
    GlobalAvgPool,
    GlobalAvgPoolGenerator,
    MaxPoolGenerator,
)

from src.modules.mbconv import *

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
    # Resnet related modules
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
]
