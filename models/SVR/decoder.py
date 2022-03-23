from typing import no_type_check
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils.model_utils import position_encoding
cudnn.benchnark=True
from models.PC.modules import *
from torch.nn import Conv1d

neg = 0.01
neg_2 = 0.2

class MyConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(MyConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

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

        self.convFeat = nn.Linear(128, n_basis, 1)
        self.projection = SubspaceLayer(dim=num_points*in_channels, n_basis=n_basis)
        self.subspace_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, h, z):
        z = self.convFeat(z)
        phi = self.projection(z).view(h.shape)
        h = h + self.subspace_conv1(phi)
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
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
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

    def forward(self, x, z):

        B,_,N = x.size()
        feat = z.unsqueeze(2).repeat(1, 1, self.args.number_points)
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