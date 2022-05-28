# encoding=utf-8

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# add for shape-preserving Loss
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
cudnn.benchnark=True
from models.PC.modules import *
from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

neg = 0.01
neg_2 = 0.2
class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = input
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x

class SP_DecoderLight(nn.Module):
    def __init__(self, args):
        super(SP_DecoderLight, self).__init__()
        self.args = args
        self.np = args.np
        self.nk = args.nk//2
        self.nz = args.nz

        Conv = nn.Conv1d
        Linear = nn.Linear

        dim = [3, 128, 512]

        self.head = nn.Sequential(
            Conv(3 + self.nz, dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.global_conv = nn.Sequential(
            Linear(dim[-1], dim[-1]),
            nn.BatchNorm1d(dim[-1]),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim[-1], 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )


        self.tail = nn.Sequential(
            Conv1d(512, 256, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(256, 64, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 3, 1),
            nn.Tanh()
        )

        self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.adain1 = AdaptivePointNorm(dim[1], dim[-1])
        self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.adain2 = AdaptivePointNorm(dim[2], dim[-1])

        self.conv_latent = nn.Conv1d(dim[-1], dim[-2], 1)


        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)

    def forward(self, x, z):

        B,_,N = x.size()

        style = torch.cat([x, z], dim=1)
        style = self.head(style)  # B,C,N

        x1 = self.EdgeConv1(x)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x1 = self.conv_latent(style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        x1_o = self.tail(x2)            

        x1_p = x1_o

        return x1_p


class SP_Decoder(nn.Module):
    def __init__(self, args):
        super(SP_Decoder, self).__init__()
        self.args = args
        self.np = args.np
        self.nk = args.nk//2
        self.nz = args.nz

        Conv = nn.Conv1d
        Linear = nn.Linear

        dim = [3, 128, 512, 1024]

        self.head = nn.Sequential(
            Conv(3 + self.nz, dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.global_conv = nn.Sequential(
            Linear(dim[-1], dim[-1]),
            nn.BatchNorm1d(dim[-1]),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim[-1], 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(neg, inplace=True),
        )


        self.tail = nn.Sequential(
            Conv1d(1024+dim[-1], 512, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(512, 128, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(128, 32, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(32, 3, 1),
            nn.Tanh()
        )

        self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.adain1 = AdaptivePointNorm(dim[1], dim[-1])
        self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.adain2 = AdaptivePointNorm(dim[2], dim[-1])
        self.EdgeConv3 = EdgeBlock(dim[2], dim[-1], self.nk)
        self.adain3 = AdaptivePointNorm(dim[-1], dim[-1])


        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)
        self.lrelu3 = nn.LeakyReLU(neg_2)



    def forward(self, x, z):

        B,N,_ = x.size()

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        x3 = self.EdgeConv3(x2)
        x3 = self.lrelu3(x3)
        x3 = self.adain3(x3, style)


        feat_global = torch.max(x3, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x3), dim=1)
        print(feat_cat.shape)

        x1_o = self.tail(feat_cat)            

        x1_p = x1_o

        return x1_p


class SP_DecoderOri(nn.Module):
    def __init__(self, args):
        super(SP_DecoderOri, self).__init__()
        self.args = args
        self.np = args.np
        self.nk = args.nk//2
        self.nz = args.nz

        Conv = nn.Conv1d
        Linear = nn.Linear

        dim = [3, 64, 128]

        self.head = nn.Sequential(
            Conv(3 + self.nz, dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.tail = nn.Sequential(
            Conv1d(128, 64, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 32, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(32, 3, 1),
            nn.Tanh()
        )

        self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.adain1 = AdaptivePointNorm(dim[1], dim[-1])
        self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.adain2 = AdaptivePointNorm(dim[2], dim[-1])

        self.conv_latent = nn.Conv1d(dim[-1], dim[-2], 1)


        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)

    def forward(self, x, z):

        B,_,N = x.size()

        style = torch.cat([x, z], dim=1)
        style = self.head(style)  # B,C,N

        x1 = self.EdgeConv1(x)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        x1_o = self.tail(x2)            

        x1_p = x1_o + x

        return x1_p


class SubspaceLayer(nn.Module):
    def __init__(self, dim, n_basis):
        super(SubspaceLayer, self).__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))    # (6,96)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))    # (6)
        self.mu = nn.Parameter(torch.zeros(dim))    # (96)

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu

class EigenBlock(nn.Module):
    def __init__(
        self,
        num_points,
        in_channels,
        n_basis
    ):
        super().__init__()

        self.convFeat1 = nn.Linear(1024, 128, 1)
        self.convFeat2 = nn.Linear(128, n_basis, 1)
        self.projection = SubspaceLayer(dim=num_points*in_channels, n_basis=n_basis)
        self.subspace_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h, z):
        z = self.convFeat1(z)
        z = self.convFeat2(z)
        phi = self.projection(z).view(h.shape)
        h = h + self.subspace_conv1(phi) * self.gamma
        return h

class SP_DecoderEigen3steps(nn.Module):
    def __init__(self, args):
        super(SP_DecoderEigen3steps, self).__init__()
        self.args = args
        self.nk = args.nk//2
        self.nz = args.nz

        Conv = nn.Conv1d

        dim = [3, 32, 64, 128]

        self.head = nn.Sequential(
            Conv(3 + self.nz, dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.tail = nn.Sequential(
            Conv1d(128, 64, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 32, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(32, 3, 1),
            nn.Tanh()
        )

        self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.adain1 = AdaptivePointNorm(dim[1], dim[-1])
        self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.adain2 = AdaptivePointNorm(dim[2], dim[-1])
        self.EdgeConv3 = EdgeBlock(dim[2], dim[3], self.nk)
        self.adain3 = AdaptivePointNorm(dim[3], dim[-1])

        self.EigenBlock1 = EigenBlock(num_points=2048, in_channels=32, n_basis=18)
        self.EigenBlock2 = EigenBlock(num_points=2048, in_channels=64, n_basis=18)
        self.EigenBlock3 = EigenBlock(num_points=2048, in_channels=128, n_basis=18)


        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)
        self.lrelu3 = nn.LeakyReLU(neg_2)

    def forward(self, x, feat, z):

        B,_,N = x.size()
        style = torch.cat([x, feat], dim=1)
        style = self.head(style)  # B,C,N

        x1 = self.EdgeConv1(x)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)
        x1 = self.EigenBlock1(x1, z)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)
        x2 = self.EigenBlock2(x2, z)

        x3 = self.EdgeConv3(x2)
        x3 = self.lrelu3(x3)
        x3 = self.adain3(x3, style)
        x3 = self.EigenBlock3(x3, z)

        x1_o = self.tail(x3)            

        x1_p = x1_o + x

        return x1_p

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)
