from models.SVR.encoder import *
from models.SVR.decoder import *
import torch
import torch.nn as nn

from models.SVR.encoder import resnet18
from utils.model_utils import get_spherepoints


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = resnet18(pretrained=False, num_classes=1000)
        self.linear = nn.Linear(1000, 128)
        self.decoder = SP_DecoderEigen3steps(args)

    def forward(self, images, sphere_xyz=None):
        
        if sphere_xyz is not None:
            sphere_points = sphere_xyz.unsqueeze(0).repeat(images.shape[0], 1, 1).cuda()
        else:
            sphere_points = get_spherepoints(2048, self.args.radius)
            sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(images.shape[0], 1, 1).cuda()    # B,3,2048
            sphere_points = sphere_points.transpose(2,1).contiguous()
        feat = self.encoder(images)
        feat = self.linear(feat)
        outputs = self.decoder(sphere_points, feat)

        return outputs.transpose(2,1).contiguous()
