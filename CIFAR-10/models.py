import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Fix this
import sys
sys.path.append("modules")
import IR2Net

class IR2Net_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, variation_range=0.1):
        super(IR2Net_ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nonlinear = nn.Hardtanh()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, variation_range=variation_range)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, variation_range=variation_range)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, variation_range=variation_range)
        self.avgpool = nn.AdaptiveAvgPool2d(8)
        self.fusion = IR2Net.CIRec(128, 64, ratio=4)   # ratio=s=4if128,s=3if80
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(IR2Net._weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, variation_range=0.1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, variation_range=variation_range))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear(self.bn1(self.conv1(x)))
        fx0 = self.avgpool(out)
        out = self.layer1(out)
        fx1 = self.avgpool(out)
        out = self.layer2(out)
        fx2 = self.avgpool(out)
        out = self.layer3(out)
        feature = out
        out = torch.cat([fx0, fx1, fx2, out], dim=1)
        out = self.fusion(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out, feature

    def update_variation(self, variation_range=0.1):
        for layer in self.layer1:
            if isinstance(layer, IR2Net.BasicBlock):
                layer.update_variation(variation_range=variation_range)
        for layer in self.layer2:
            if isinstance(layer, IR2Net.BasicBlock):
                layer.update_variation(variation_range=variation_range)
        for layer in self.layer3:
            if isinstance(layer, IR2Net.BasicBlock):
                layer.update_variation(variation_range=variation_range)

def IR2Net_ResNet20(variation_range=0.1):
    return IR2Net_ResNet(IR2Net.BasicBlock, [3, 3, 3], variation_range=variation_range)
