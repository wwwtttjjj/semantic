import imp
from statistics import mode
from turtle import forward

from zmq import IPC_FILTER_GID
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision



class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classifier):
        super(_SimpleSegmentationModel,self).__init__() 
        self.backbone = backbone
        self.num_classifier = num_classifier
    def forward(self, x):
        input_shape = x.shape[-2:]#h and w, dont have channels
        x = self.backbone(x)
        x = self.num_classifier(x)
        x = F.interpolate(x, input_shape, mode = 'bilinear', align_corners=False)
        return x


class ConvBatchRelu(nn.Module):
    def __init__(self, input, output, kernel_size,padding,dialtion_rate):
        super(ConvBatchRelu, self).__init__()
        self.CBR = nn.Sequential(nn.Conv2d(input, output, kernel_size,stride=1,padding=padding, dilation = dialtion_rate),
                                    nn.BatchNorm2d(output),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        return self.CBR(x)

class MobileNetv2(nn.Module):
    """
    Moblienet backbone
    input: outstride
    """
    def __init__(self, outstride):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.features = self.model.features[:-1]
        if outstride == 16:
            inverted_residual = self.features[14]
            conv = inverted_residual.conv[1][0]
            conv.stride = (1, 1)
            conv.dilation = (2, 2)
            conv.padding = (2, 2)
        elif outstride == 8:
            conv_0 = self.features[7].conv[1][0]
            conv_0.stride = (1, 1)
            conv_0.dilation = (2, 2)
            conv_0.padding = (2, 2)
            conv_1 = self.features[14].conv[1][0]
            conv_1.stride = (1, 1)
            conv_1.dilation = (4, 4)
            conv_1.padding = (4, 4)
    def forward(self,x):
        feature0 = self.features[0:4](x)
        feature1 = self.features[4:](feature0)

        return feature0, feature1


class ASPP(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.CBR0 = ConvBatchRelu(input, output, 1, padding=0,dialtion_rate=1)
        self.CBR1 = ConvBatchRelu(input, output, 3, padding=6,dialtion_rate=6)
        self.CBR2 = ConvBatchRelu(input, output, 3, padding=12,dialtion_rate=12)
        self.CBR3 = ConvBatchRelu(input, output, 3, padding=18,dialtion_rate=18)
        self.CBR4 = ConvBatchRelu(input, output, 1, padding=0,dialtion_rate=1)
        self.CBR5 = ConvBatchRelu(output * 5, output, 1, padding=0,dialtion_rate=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self,x):
        h, w = x.shape[:-2]
        x1 = self.CBR0(x)
        x2 = self.CBR1(x)
        x3 = self.CBR2(x)
        x4 = self.CBR3(x)
        x5 = F.interpolate(self.pooling(self.CBR4(x)),(h, w), mode = 'bilinear')
        x = torch.cat([x1, x2, x3, x4, x5], dim = 1)
        return self.CBR5(x)



class DeepLabV3Plus(nn.Module):
    def __init__(self, outstride, num_classes) -> None:
        super().__init__()
        self.backbone = MobileNetv2(outstride=outstride)
        self.conv0 = ConvBatchRelu(24, 48, 1, 0, 1)
        self.aspp = ASPP(320, 256)
        self.decode1 = nn.Sequential(
            ConvBatchRelu(304, 256, 3, 1, 1),
            nn.Dropout(),
            ConvBatchRelu(256, 256, 3, 1, 1),
            nn.Dropout(0.1)
        )
        self.decode2 = nn.Conv2d(256, num_classes, 1)
    def forward(self,x):
        x1, x2 = self.backbone(x)
        x3 = self.decode1(x1)
        x4 = F.interpolate(self.aspp(x2), scale_factor=4, mode='bilinear')
        x5 = torch.cat([x3, x4], dim = 1)
        x6 = self.decode2(self.decode1(x5))
        x7 = F.interpolate(x6, scale_factor=4,mode = 'bilinear')

        return x7