import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function,Variable
from scipy.stats import ortho_group

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.epoch=-1

        w=self.weight
        self.Rweight=w.clone()
        self.a,self.b=get_ab(np.prod(w.shape[1:]))
        self.R1=torch.cuda.FloatTensor(ortho_group.rvs(dim=self.a))
        self.R2=torch.cuda.FloatTensor(ortho_group.rvs(dim=self.b))
        # self.sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1, 1).detach()
        # self.alpha=nn.Parameter(torch.FloatTensor(self.sw),requires_grad=True)
        # self.rand = torch.rand(self.sw.shape).cuda()

    def forward(self, input):
        w = self.weight
        a,b = self.a,self.b
        X=w.view(w.shape[0],a,b)
        if self.epoch%2==0:
            with torch.no_grad():
                for _ in range(3):
                    #* update B
                    V = self.R1.t()@X@self.R2
                    B = torch.sign(V)
                    #* update R1
                    D1=sum([Bi@(self.R2.t())@(Xi.t()) for (Bi,Xi) in zip(B,X)]).cpu()
                    U1,S1,V1=torch.svd(D1)
                    self.R1=(V1@(U1.t())).cuda()
                    #* update R2
                    D2=sum([(Xi.t())@self.R1@Bi for (Xi,Bi) in zip(X,B)]).cpu()
                    U2,S2,V2=torch.svd(D2)
                    self.R2=(U2@(V2.t())).cuda()
        self.Rweight=nn.Parameter((self.R1.t()@X@self.R2).view_as(w))
        #* bn
        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        #* scaling factor
        # sw = bw.abs().view(bw.size(0), -1).mean(-1).float().view(bw.size(0), 1, 1, 1).detach()

        bw = BinaryQuantize().apply(self.Rweight, self.k, self.t)
        ba = BinaryQuantize().apply(input, self.k, self.t)
        # bw = bw * sw
        #bw = bw * self.alpha
        #bw = bw * self.rand 
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
        # grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        # grad_input = grad_output.clone()
        grad_input = grad_output.clone().clamp(-1,1)
        return grad_input, None, None

class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input) 
        # out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = grad_output.clone().clamp(-1.,1.)
        # grad_input[torch.add(grad_input<-1,grad_input>1)]=0
        # grad_input = grad_output.clone()
        return grad_input, None, None

def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt,0,-1):
        if N%i==0:
            return i,N//i 