import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function,Variable

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

        w = self.weight
        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1, 1).detach()
        self.alpha=nn.Parameter(torch.FloatTensor(sw),requires_grad=True)
        # self.rand = torch.rand(self.sw.shape).cuda()*0.01

    def forward(self, input):
        w = self.weight
        a = input
        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        #ba = a - a.view(a.size(0), -1).mean(-1).view(a.size(0), 1, 1, 1)
        # sw = bw.abs().view(bw.size(0), -1).mean(-1).float().view(bw.size(0), 1, 1, 1).detach()
        #* sa
        #sa = a.abs().view(a.size(0), -1).mean(-1).float().view(a.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(w, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        # bw = bw * sw
        #* sa
        #ba = ba * sa
        bw = bw * self.alpha
        # bw = bw * self.rand 
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

        
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t) 
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        # grad_input = grad_output.clone()
        # grad_input = grad_output.clone().clamp(-1,1)
        return grad_input, None, None

class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input) 
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = grad_output.clone().clamp(-1.,1.)
        # grad_input[torch.add(grad_input<-1,grad_input>1)]=0
        # grad_input = grad_output.clone()
        return grad_input, None, None
