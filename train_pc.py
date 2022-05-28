import torch.optim as optim
import torch
from utils.train_utils import *
import logging
import math
import importlib
import random
import munch
import yaml
import os
import sys
import argparse
from dataset_pc.dataset import MVP_CP
from tqdm import tqdm
from time import time
import time as timetmp

import warnings
warnings.filterwarnings("ignore")


def setFolders(args):

    LOG_DIR = args.dir_outpath
    MODEL_NAME = '%s-%s'%(args.model_name, timetmp.strftime("%m%d_%H%M", timetmp.localtime()))

    OUT_DIR = os.path.join(LOG_DIR, MODEL_NAME)
    args.dir_checkpoints = os.path.join(OUT_DIR, 'checkpoints')
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(args.dir_checkpoints):
        os.makedirs(args.dir_checkpoints)


    os.system('cp -r models %s' % (OUT_DIR))
    os.system('cp train.py %s' % (OUT_DIR))

    os.system('cp -r cfgs %s' % (OUT_DIR))

    LOG_FOUT = open(os.path.join(OUT_DIR, 'log_%s.csv' %(MODEL_NAME)), 'w')
    return MODEL_NAME, OUT_DIR, LOG_FOUT

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def train():
    # Set up folders for logs and checkpoints
    exp_name, log_dir, LOG_FOUT = setFolders(args)

    log_string('EPOCH,CD_S1,CD_S2,BEST CD', LOG_FOUT)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
                                                      
    logging.info(str(args))

    metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataset = MVP_CP(prefix="train")
    dataset_test = MVP_CP(prefix="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

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


    lr = args.lr

    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))
    optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    for epoch in range(args.start_epoch, args.nepoch):
        epoch_start_time = time()
        total_cd_step1 = 0
        total_cd_step2 = 0

        train_loss_meter.reset()
        net.module.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        n_batches = len(dataloader)
        with tqdm(dataloader) as t:
            for batch_idx, data in enumerate(t):
                optimizer.zero_grad()

                label, inputs, gt = data # [b] [b, n, 3] [b, n, 3]
                label = label.float().cuda()
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                batch_size = inputs.shape[0]

                out2, loss1, loss2, net_loss = net(inputs, gt, label=label, alpha=alpha)
                reg_loss = net.module.step1.decoder.orthogonal_regularizer() * 100
                net_loss = net_loss.mean()
                net_loss_all = net_loss + reg_loss

                train_loss_meter.update(net_loss.item())
                net_loss_all.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())

                optimizer.step()

                cd_step1_item = torch.sum(loss1).item() / batch_size * 1e4
                total_cd_step1 += cd_step1_item
                cd_step2_item = torch.sum(loss2).item() / batch_size * 1e4
                total_cd_step2 += cd_step2_item

                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch, args.nepoch, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_step1_item, cd_step2_item]])
        
        avg_cd_step1 = total_cd_step1 / n_batches
        avg_cd_step2 = total_cd_step2 / n_batches
        
        epoch_end_time = time()
        logging.info(' ')
        logging.info(
            exp_name + '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch, args.nepoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd_step1, avg_cd_step2]]))

        if epoch % args.epoch_interval_to_save == 0:
            save_model(str(log_dir) + '/checkpoints/' + str(epoch) + 'network.pth', net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir):
    metrics_val = ['cd_p', 'cd_t', 'f1', 'cd_raw_t']
    val_loss_meters = {m: AverageValueMeter() for m in metrics_val}
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            label, inputs, gt = data
            # mean_feature = None
            curr_batch_size = gt.shape[0]

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            label = label.float().cuda()

            result_dict = net(inputs, gt, label=label, prefix="val")
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item(), curr_batch_size)

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        log_string('%d,%.2f,%.2f,%.2f'%(curr_epoch_num, val_loss_meters['cd_raw_t'].avg*1e4, val_loss_meters['cd_t'].avg*1e4, best_epoch_losses['cd_t'][1]*1e4), LOG_FOUT)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)


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

    train()



