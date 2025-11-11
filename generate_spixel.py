import os
import sys
import cv2
import argparse
from tqdm import tqdm
from glob import glob
from imageio import imread, imsave
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append("./CDSpixel")
from CDSpixel import models, flow_transforms
from CDSpixel.train_util import update_spixel_map, get_spixel_image, shift9pos


path_file = {
    'pascal': './spixel_masks/path_files/pascal.txt',
    'fss': './spixel_masks/path_files/fss.txt',
    'deepglobe': './spixel_masks/path_files/deepglobe.txt',
    'isic': './spixel_masks/path_files/isic.txt',
    'lung': './spixel_masks/path_files/lung.txt',
}

data_suffix = {
    'pascal': 'jpg',
    'fss': 'jpg',
    'deepglobe': 'jpg',
    'isic': 'jpg',
    'lung': 'png'
}

spixel_size = [5, 10, 15, 20]

input_transform = transforms.Compose([
    flow_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
    transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
])


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='fss', choices=['pascal', 'fss', 'deepglobe', 'isic', 'lung'], 
                                     help='validation dataset')
    parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                                        default= './CDSpixel/weights/model_best.tar')
    parser.add_argument('--downsize', default=16, type=float,help='superpixel grid cell, must be same as training setting')
    parser.add_argument('-nw', '--num_threads', default=1, type=int,  help='num_threads')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

    args = parser.parse_args()
    return args


@torch.no_grad()
def extract_spixel_mask(args, model, img_path, save_path):
    image_list = sorted(glob(os.path.join(img_path, f"*.{data_suffix[args.dataset]}")))
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(len(image_list))):
        img_file = image_list[i]
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        save_mask_file = os.path.join(save_path, img_name + '.npy')

        img = imread(img_file)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[:, :, :3]

        spixel_map_list = []
        for size in spixel_size:
            H, W = int(size*16), int(size*16)

            img1 = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
            img1 = input_transform(img1)

            # compute output
            output = model(img1.cuda().unsqueeze(0), )

            # get spixel id
            n_spixl_h = int(np.floor(H / args.downsize))
            n_spixl_w = int(np.floor(W / args.downsize))
            n_spixel =  int(n_spixl_h * n_spixl_w)
            # [nh,nw]
            spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
            # [9,nh,nw]
            spix_idx_tensor_ = shift9pos(spix_values)
            spix_idx_tensor = np.repeat(
            np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)
            # [1,9,H,W]
            spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

            # assign the spixel map
            curr_spixl_map = update_spixel_map(spixeIds, output)
            ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H,W), mode='nearest').type(torch.int)

            # mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
            # spixel_viz, spixel_map = get_spixel_image((img1 + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels=n_spixel, b_enforce_connect=False)

            if not isinstance(ori_sz_spixel_map.squeeze(), np.ndarray):
                spixel_map = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0,1)
            else:
                spixel_map = ori_sz_spixel_map.squeeze()

            # resize, [H,W,1], [1, spixel_size[idx]^2]
            spixel_map = cv2.resize((spixel_map+1).astype(int), (spixel_size[-1]*16, spixel_size[-1]*16), interpolation=cv2.INTER_NEAREST)
            spixel_map = np.expand_dims(spixel_map, axis=-1)
            spixel_map_list.append(spixel_map)
        
        # [H,W,N]
        spixel_map = np.concatenate(spixel_map_list, axis=-1)
        np.save(save_mask_file, spixel_map)


def read_path_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


if __name__ == '__main__':
    args = parse_args()

    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    img_paths = read_path_file(path_file[args.dataset])

    for img_path in img_paths:
        print(img_path)
        save_path = img_path.replace("datasets", "spixel_masks", 1)
        extract_spixel_mask(args, model, img_path, save_path)