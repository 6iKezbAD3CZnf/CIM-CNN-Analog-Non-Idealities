import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function

# TODO: Fix this
import sys
sys.path.append("modules")
import IR2Net

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.layers(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

class MisMatchedLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, variation, bias=None):
        ctx.save_for_backward(input, weight, bias)
        return F.linear(input, weight + variation, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            raise RuntimeError("Variations do not require grad")
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, None, grad_bias

class MisMatchedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, variation_range=0.1):
        super(MisMatchedLinear, self).__init__(in_features, out_features, bias)

        self.variation_range = variation_range
        variation = torch.rand_like(self.weight, requires_grad=False) * self.variation_range * 2 - self.variation_range
        self.register_buffer('variation', variation)

        self.weight.data -= self.variation

    def forward(self, input):
        return MisMatchedLinearFunction.apply(input, self.weight, self.variation, self.bias)

    def update_variation(self, variation_range=0.1):
        self.variation_range = variation_range
        self.variation = torch.rand_like(self.weight, requires_grad=False) * self.variation_range * 2 - self.variation_range

class MisMatchedReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, variation):
        ctx.save_for_backward(input)
        return F.relu(input + variation)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
        return grad_input, None

class MisMatchedReLU(nn.Module):
    def __init__(self, variation_range=1.0):
        super(MisMatchedReLU, self).__init__()
        self.variation_range = variation_range
        self.variation = None

    def forward(self, input):
        if self.variation is None or self.variation.shape != input.shape:
            self.variation = torch.rand_like(input, requires_grad=False) * self.variation_range * 2 - self.variation_range
        return MisMatchedReLUFunction.apply(input, self.variation)

    def update_variation(self, variation_range=1.0):
        self.variation_range = variation_range
        self.variation = None

class MisMatchedLeNet(nn.Module):
    def __init__(self, linear_variation_range=0.1, relu_offset_range=1.0):
        super(MisMatchedLeNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            MisMatchedReLU(variation_range=relu_offset_range),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            MisMatchedReLU(variation_range=relu_offset_range),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            MisMatchedLinear(256, 120, bias=True, variation_range=linear_variation_range),
            MisMatchedReLU(variation_range=relu_offset_range),
            MisMatchedLinear(120, 84, bias=True, variation_range=linear_variation_range),
            MisMatchedReLU(variation_range=relu_offset_range),
            MisMatchedLinear(84, 10, bias=True, variation_range=linear_variation_range),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

    def update_variation(self, linear_variation_range=1.0, relu_offset_range=0.1):
        for layer in self.layers:
            if isinstance(layer, MisMatchedLinear):
                layer.update_variation(linear_variation_range)
            elif isinstance(layer, MisMatchedReLU):
                layer.update_variation(relu_offset_range)
