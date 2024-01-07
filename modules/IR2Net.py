import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function

def _weights_init(m):
    classname = m.__class__.__name__
    #  print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.binary_activation = BinaryActivation()

    def forward(self, input):
        w = self.weight
        a = input

        binary_weights_no_grad = torch.sign(w)
        cliped_weights = torch.clamp(w, -1.0, 1.0)
        bw = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        ba = self.binary_activation(a)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output

class OffsetVariationFunction(Function):
    @staticmethod
    def forward(ctx, input, offsets):
        # Calculate the repeat factor
        repeat_factor = (input.numel() + 511) // 512
        repeated_offsets = offsets.repeat(repeat_factor)[:input.numel()].reshape(input.shape)
        return input + repeated_offsets

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class OffsetVariation(nn.Module):
    def __init__(self, num_offsets=512, offset_range=0.1):
        super(OffsetVariation, self).__init__()

        # Initialize offsets
        self.offset_range = offset_range
        self.offsets = nn.Parameter(
                torch.rand(num_offsets) * self.offset_range * 2 - self.offset_range,
                requires_grad=False
                )

    def forward(self, input):
        return OffsetVariationFunction.apply(input, self.offsets)

    def update_offset(self, offset_range=0.1):
        self.offset_range = offset_range
        self.offsets = nn.Parameter(
                torch.rand_like(self.offsets) * self.offset_range * 2 - self.offset_range,
                requires_grad=False
                )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', variation_range=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ov1 = OffsetVariation(offset_range=variation_range)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ov2 = OffsetVariation(offset_range=variation_range)
        self.shortcut = nn.Sequential()
        self.nonlinear = nn.Hardtanh()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                                   F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = self.ov1(out)
        out = self.nonlinear(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = self.ov2(out)
        out = self.nonlinear(out)
        return out

    def update_variation(self, variation_range=0.1):
        self.ov1.update_offset(offset_range=variation_range)
        self.ov2.update_offset(offset_range=variation_range)

class CIRec(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(CIRec, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)   # m=init_channels, ratio=s, oup=n
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Hardtanh(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.Hardtanh(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
