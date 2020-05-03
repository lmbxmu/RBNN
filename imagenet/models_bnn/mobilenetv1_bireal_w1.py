'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *

__all__=['mobilenetv1_bireal_1w1a','mobilenetv1_bireal_025_1w1a','mobilenetv1_bireal_05_1w1a']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,alpha=1):
        super(Block, self).__init__()
        self.first = True if in_planes==32  else False 
        if in_planes == out_planes: #* concat or not
            self.concat = False 
        else: 
            self.concat = True 
            out_planes = in_planes

        self.conv1 = BinarizeConv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        if self.concat:
            self.conv3 = BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)
        self.Hardtanh=nn.Hardtanh()

    def forward(self, x):
        if self.concat and not self.first:
            shortcut = F.avg_pool2d(x,2)
        else: 
            shortcut = x 
        out = self.bn1(self.conv1(x))
        out += shortcut
        out=self.Hardtanh(out)
        
        shortcut = out 
        out1 = self.bn2(self.conv2(out))
        out1 += shortcut 
        out1 = self.Hardtanh(out1)

        if self.concat:
            out2 = self.bn3(self.conv3(out))
            out2 += shortcut
            out2 = self.Hardtanh(out2)
            out = torch.cat((out1,out2),1)
        else: 
            out = out1 
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10,alpha=1):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32,alpha=alpha)
        self.Hardtanh=nn.Hardtanh()
        self.linear = nn.Linear(int(1024*alpha), num_classes)
        #* weight_init
        self.apply(_weights_init)

    def _make_layers(self, in_planes,alpha):
        layers = []
        if alpha!=1:
            self.cfg = self.cfg[int(1/alpha):]
        for x in self.cfg:
            out_planes = int(x * alpha) if isinstance(x, int) else int(x[0] * alpha)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride,alpha))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.Hardtanh(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenetv1_bireal_1w1a():
    return MobileNet() 

def mobilenetv1_bireal_025_1w1a():
    return MobileNet(alpha=0.25)
    
def mobilenetv1_bireal_05_1w1a():
    return MobileNet(alpha=0.5) 
    
def mobilenetv1_bireal_075_1w1a():
    return MobileNet(alpha=0.75) 

def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
