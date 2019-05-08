import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * (self.relu6(x+3)) / 6

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return (self.relu6(x+3)) / 6

ACT_FNS = {
    'RE': nn.ReLU6(inplace=True),
    'HS': HardSwish(),
    'HG': HardSigmoid()
}

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride, nl='RE'):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        ACT_FNS[nl]
    )

def conv_1x1(inp, oup, nl='RE', with_se=False):
    if with_se:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            SqueezeAndExcite(oup, reduction=4),
            ACT_FNS[nl]
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ACT_FNS[nl]
        )

def conv_1x1_bn(inp, oup, nl='RE', with_se=False):
    if with_se:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            SqueezeAndExcite(oup, reduction=4),
            nn.BatchNorm2d(oup),
            ACT_FNS[nl]
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ACT_FNS[nl]
        )

class SqueezeAndExcite(nn.Module):
    def __init__(self, n_features, reduction=4):
        super(SqueezeAndExcite, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = ACT_FNS['RE']
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = ACT_FNS['HG']

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_size, nl='RE', with_se=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expand_size

        self.identity = stride == 1 and inp == oup

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            ACT_FNS[nl],
        )

        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            ACT_FNS[nl],
        )

        self.se = nn.Sequential(
            SqueezeAndExcite(hidden_dim, reduction=4)
        )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        if with_se: # with squeeze and excite 
            if expand_size == oup: # exp_ratio = 1
                self.conv = nn.Sequential(
                    self.dw,
                    self.se,
                    self.pw_linear,
                )
            else:
                self.conv = nn.Sequential(
                    self.pw,
                    self.dw,
                    self.se,
                    self.pw_linear,
                )
        else:
            if expand_size == oup:
                self.conv = nn.Sequential(
                    self.dw,
                    self.pw_linear,
                )
            else:
                self.conv = nn.Sequential(
                    self.pw,
                    self.dw,
                    self.pw_linear,
                )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):

    # NOTE: [kernel, expansion, output, SE, NL, s]
    cfg = [(3,  16, 16, True,  'RE', 2),
           (3,  72, 24, False, 'RE', 2),  
           (5,  88, 24, False, 'RE', 1),
           (5,  96, 40, True,  'HS', 2),
           (5, 240, 40, True,  'HS', 1),
           (5, 240, 40, True,  'HS', 1),
           (5, 120, 48, True,  'HS', 1),
           (5, 144, 48, True,  'HS', 1),
           (5, 288, 96, True,  'HS', 2),
           (5, 576, 96, True,  'HS', 1),
           (5, 576, 96, True,  'HS', 1)]

    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # building first layer
        assert input_size % 32 == 0
        input_channel = _make_divisible(16 * width_mult, 8)
        self.conv0 = conv_3x3_bn(3, input_channel, 2, nl='HS')
        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        # for t, c, n, s in self.cfgs:
        for kernel, expansion, output_channel, se, nl, stride in self.cfg:
            layers.append(block(input_channel, output_channel, kernel, stride, expansion, nl=nl, with_se=se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        self.conv1 = conv_1x1_bn(input_channel, expansion, nl='HS', with_se=False)
        input_channel = expansion

        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1(input_channel, output_channel, nl='HS', with_se=False)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv0(x)
        x = self.features(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = MobileNetV3()
    x_image = Variable(torch.randn(1, 3, 224, 224))
    y = net(x_image)
