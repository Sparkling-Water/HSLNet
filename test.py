import os
import argparse
from tqdm import tqdm

import torch
from torch.nn import DataParallel

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
                        default='fss',
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
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101', 'ViT-B/16-384', 'vit_b', 'dino'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--perturb',
                        action='store_true',
                        default=False,
                        help='whether to use domain perturbation')

    parser.add_argument("--checkpoint",
                    type=str,
                    default= "./output/vit_b/1shot/best/best_model.pth")

    args = parser.parse_args()
    return args


def evaluate(model, dataloader):
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
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
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=10)

    average_meter.write_result()

    return


if __name__ == '__main__':
    args = parse_args()
    args.output_path = os.path.dirname(args.checkpoint)
    
    Logger.initialize(args, mode='test')

    FSSDataset.initialize(img_size=args.size, datapath=args.datapath)
    testloader = FSSDataset.build_dataloader(args.dataset, 1, 0, 0, 'test', args.shot)

    model = HSLNet(args)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model)
    model.to(device)
    model.eval()

    utils.fix_randseed(args.seed)
    evaluate(model, testloader, args)

    Logger.info('==================== Finished Testing ====================')
