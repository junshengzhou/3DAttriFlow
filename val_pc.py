import argparse
import logging
import os
import munch
import torch
import yaml
from tqdm import tqdm
import numpy as np

import logging
import torch
from tqdm import tqdm
from dataset_pc.dataset import MVP_CP
from utils.model_utils import calc_cd
import argparse
import torch
import logging
import importlib
import random
import munch
import yaml
import os
import sys
import argparse
from dataset_pc.dataset import MVP_CP
import numpy as np
from tqdm import tqdm
from time import time
import time as timetmp
from utils.model_utils import *
from utils.train_utils import AverageValueMeter

def val():
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    test_dataset_loader = MVP_CP(prefix="test")
    test_data_loader = torch.utils.data.DataLoader(test_dataset_loader, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
    logging.info('Length of test dataset:%d', len(test_dataset_loader))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    
    ckpt = torch.load(args.load_model)
    net.module.load_state_dict(ckpt['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)

    # Switch models to evaluation mode
    net.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    test_losses = AverageValueMeter()

    with tqdm(test_data_loader) as t:
        for model_idx, data in enumerate(t):

            with torch.no_grad():

                category, label, partial, gt = data
                partial = partial.float().cuda()  # B, 2048, 3
                gt = gt.float().cuda()  # B, 2048, 3
                label = label.float().cuda()

                _, _, cd_t, _ = net(partial, gt, label)

                cd_t = cd_t.mean().item() * 1e4

                test_losses.update(cd_t, partial.shape[0])

                t.set_description('Test[%d/%d] Metrics = %.4f' %
                             (model_idx + 1, n_samples, cd_t))

    print('============================ TEST RESULTS ============================')

    print('Overall cd: ', (test_losses.avg))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    print('Using gpu:' + str(arg.gpu_id))

    val()