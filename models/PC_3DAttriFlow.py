import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.PC.decoder import SP_DecoderEigen3steps
from models.PC.re_module import RE_Module
from utils.model_utils import *

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2

from models.PC.encoder import PointsEncoder_pointwise
import torch
import torch.nn as nn
from utils.model_utils import get_spherepoints

class ModelStep1(nn.Module):
    def __init__(self, args):
        super(ModelStep1, self).__init__()
        self.args = args
        self.encoder = PointsEncoder_pointwise()
        self.decoder = SP_DecoderEigen3steps(args)

    def forward(self, points, label, sphere_xyz=None):
        if sphere_xyz is not None:
            sphere_points = sphere_xyz.unsqueeze(0).repeat(points.shape[0], 1, 1).cuda()
        else:
            sphere_points = get_spherepoints(2048, self.args.radius)
            sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(points.shape[0], 1, 1).cuda()  
            sphere_points = sphere_points.transpose(2,1).contiguous()

        feat, g_feat = self.encoder(points, label)   
        outputs = self.decoder(sphere_points, feat, g_feat)

        return outputs

class Model(nn.Module):
    def __init__(self, args, size_z=128):
        super(Model, self).__init__()

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]

        self.step1 = ModelStep1(args)
        self.size_z = size_z
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.decoder = RE_Module(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse,
                                        num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list,
                                        pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)

    def forward(self, x, gt=None, label=None, prefix="train", alpha=0):

        x = x.transpose(1, 2).contiguous()
        coarse_raw = self.step1(x, label)
        _, coarse_high, coarse, fine = self.decoder(coarse_raw, x)  
        coarse_raw = coarse_raw.transpose(1, 2).contiguous()
        coarse_high = coarse_high.transpose(1, 2).contiguous()
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()

        if prefix=="train":
            if self.train_loss == 'cd':
                loss1, loss1_t = calc_cd(coarse_raw, gt)
                loss2, loss2_t = calc_cd(coarse_high, gt)
                loss3, loss3_t = calc_cd(coarse, gt)
                loss4, loss4_t = calc_cd(fine, gt)
            else:
                raise NotImplementedError('Only CD is supported')

            total_train_loss = loss1.mean() * 10 + loss2.mean() * 0.5 + loss3.mean() + loss4.mean() * alpha
            return fine, loss1_t, loss4_t, total_train_loss
        elif prefix=="val":
            if self.eval_emd:
                emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
            cd_raw_p, cd_raw_t = calc_cd(coarse_raw, gt)
            return {'out1': coarse_raw, 'out2': fine, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1, 'cd_raw_t':cd_raw_t}
