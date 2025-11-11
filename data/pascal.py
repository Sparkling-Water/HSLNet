r""" PASCAL-5i few-shot semantic segmentation dataset """
import os
import cv2

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torchvision import transforms
import PIL.Image as Image
import numpy as np


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        # if fold_id > nflods, use all folds
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.img_path = os.path.join(datapath, 'PASCAL/VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'PASCAL/VOCdevkit/VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize, query_spixel_mask, support_spixel_masks = self.load_frame(query_name, support_names)

        # [C,H,W]
        query_spixel_mask_list = []
        for i in range(query_spixel_mask.shape[-1]):
            query_spixel_mask_list.append(cv2.resize(query_spixel_mask[:, :, i], (query_cmask.width, query_cmask.height), interpolation=cv2.INTER_NEAREST))
        query_spixel_mask = np.stack(query_spixel_mask_list, axis=-1)
        query_img, query_cmask, query_spixel_mask, _ = self.transform(query_img, query_cmask, query_spixel_mask)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        # [N,C,H,W]
        support_masks = []
        support_ignore_idxs = []
        for i in range(len(support_imgs)):
            support_spixel_mask_list = []
            for j in range(support_spixel_masks[i].shape[-1]):
                support_spixel_mask_list.append(cv2.resize(support_spixel_masks[i][:, :, j], (support_cmasks[i].width, support_cmasks[i].height), interpolation=cv2.INTER_NEAREST))
            support_spixel_mask = np.stack(support_spixel_mask_list, axis=-1)
            support_imgs[i], support_cmasks[i], support_spixel_masks[i], _ = self.transform(support_imgs[i], support_cmasks[i], support_spixel_mask)
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmasks[i], class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)

        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        # [L,H,W]
        query_spixel_mask = query_spixel_mask.permute(2, 0, 1)
        support_spixel_masks = torch.stack(support_spixel_masks, dim=0).permute(0, 3, 1, 2)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'query_spixel_mask': query_spixel_mask,
                 'support_spixel_masks': support_spixel_masks,           

                 'class_id': torch.tensor(class_sample)}
        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        query_spixel_mask = self.get_spixel_mask(os.path.join(self.img_path.replace('datasets', 'spixel_masks'), query_name))
        support_spixel_masks = [self.get_spixel_mask(os.path.join(self.img_path.replace('datasets', 'spixel_masks'), supp_name))
                            for supp_name in support_names]

        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize, query_spixel_mask, support_spixel_masks

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
    
    def get_spixel_mask(self, mask_name):
        mask_name = mask_name +  '.npy'
        spixel_mask = np.load(mask_name)
        return spixel_mask

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: 
                support_names.append(support_name)
            if len(support_names) == self.shot: 
                break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # [[name,class_id],...]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            if self.fold != 4:
                img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata


    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        # {'0':[img_name1,img_name2,...],'1':[img_name3,img_name4,...],...}
        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
