import os
import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import SGD

from data.dataset import FSSDataset
from model.hslnet import HSLNet
from common import utils
from common.evaluation import Evaluator
from common.logger import Logger, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation')


    parser.add_argument('--output-path',
                        type=str,
                        default='./output',
                        help='The path of output')

    parser.add_argument('--datapath',
                        type=str,
                        default="./datasets",
                        help='The path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='lung',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='validation dataset')
    parser.add_argument('--size',
                        type=int,
                        default=400,
                        help='Size of training samples')
    
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--episode',
                        type=int,
                        default=6000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'vit_b'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    
    parser.add_argument('--perturb',
                        action='store_true',
                        default=False,
                        help='whether to use domain perturbation')

    args = parser.parse_args()
    return args


def evaluate(model, dataloader):
    tbar = tqdm(dataloader)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(tbar):
        batch = utils.to_cuda(batch)

        # [B,K,C,H,W] -> [K,B,C,H,W]
        img_s = batch['support_imgs'].permute(1,0,2,3,4)
        # [B,K,H,W] -> [K,B,H,W]
        mask_s = batch['support_masks'].permute(1,0,2,3)
        # [B,C,H,W]
        img_q = batch['query_img']
        # [B,H,W]
        mask_q = batch['query_mask']
        # spixel mask, [B,L,H,W]
        spixel_mask_q = batch['query_spixel_mask']
        # spixel mask, [B,K,L,H,W] -> [K,B,L,H,W]
        spixel_mask_s = batch['support_spixel_masks'].permute(1,0,2,3,4)

        img_s_list = [img_s[k] for k in range(img_s.shape[0])]
        mask_s_list = [mask_s[k] for k in range(mask_s.shape[0])]
        spixel_mask_s_list = [spixel_mask_s[k] for k in range(spixel_mask_s.shape[0])]

        with torch.no_grad():
            pred_mask, _ = model(img_q, mask_q, img_s_list, mask_s_list, spixel_mask_q, spixel_mask_s_list)

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        miou, fb_iou = average_meter.compute_iou()

        tbar.set_description("Testing mIOU: %.2f" % (miou))

    average_meter.write_result()

    return miou, fb_iou


if __name__ == '__main__':
    args = parse_args()
    
    Logger.initialize(args)

    FSSDataset.initialize(img_size=args.size, datapath=args.datapath)
    trainloader = FSSDataset.build_dataloader('pascal', args.batch_size, 4, 4, 'trn', args.shot)
    FSSDataset.initialize(img_size=args.size, datapath=args.datapath)
    valloader = FSSDataset.build_dataloader(args.dataset, 1, 4, 0, 'test', args.shot)

    model = HSLNet(args)
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model)
    model.to(device)

    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    previous_best = 0
    best_model = None
    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        Logger.info(f'\nEpoch {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, Previous best: {previous_best}')

        model.train()
        # for module in model.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.eval()

        total_loss = 0.0
        tbar = tqdm(trainloader)
        utils.fix_randseed(None)

        for idx, batch in enumerate(tbar):
            batch = utils.to_cuda(batch)

            # [B,K,C,H,W] -> [K,B,C,H,W]
            img_s = batch['support_imgs'].permute(1,0,2,3,4)
            # [B,K,H,W] -> [K,B,H,W]
            mask_s = batch['support_masks'].permute(1,0,2,3)
            # [B,C,H,W]
            img_q = batch['query_img']
            # [B,H,W]
            mask_q = batch['query_mask']
            # spixel mask, [B,L,H,W]
            spixel_mask_q = batch['query_spixel_mask']
            # spixel mask, [B,K,L,H,W] -> [K,B,L,H,W]
            spixel_mask_s = batch['support_spixel_masks'].permute(1,0,2,3,4)

            img_s_list = [img_s[k] for k in range(img_s.shape[0])]
            mask_s_list = [mask_s[k] for k in range(mask_s.shape[0])]
            spixel_mask_s_list = [spixel_mask_s[k] for k in range(spixel_mask_s.shape[0])]

            pred_mask, out_ls = model(img_q, mask_q, img_s_list, mask_s_list, spixel_mask_q, spixel_mask_s_list)

            loss = model.module.compute_loss(out_ls, mask_q, mask_s_list)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            tbar.set_description('Loss: %.3f' % (total_loss / (idx + 1)))
        
        # evaluate
        model.eval()
        utils.fix_randseed(args.seed)
        miou, fb_iou = evaluate(model, valloader, args)
        torch.cuda.empty_cache()

        # best model
        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            Logger.save_model_miou(best_model, epoch, miou)
        
        torch.save(model.module.state_dict(), os.path.join(Logger.logpath, f'epoch{epoch}_model.pth'))
    
    Logger.info('==================== Finished Training ====================')
 