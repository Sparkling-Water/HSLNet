import os
import sys
import copy

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from torch.hub import download_url_to_file

import model.base.resnet as resnet
import model.base.vit as vit


# resnet backbone
class Backbone_Res(nn.Module):
    def __init__(self, backbone):
        super(Backbone_Res, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.backbone = nn.ModuleList([self.layer0, self.layer1, self.layer2, self.layer3])

        self.proj_layer = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, img_q, img_s):
        B, K, C, H, W = img_s.shape

        # [B，K+1, C, H, W]
        img_cat = torch.cat((img_s, img_q.view(B, 1, C, H, W)), dim=1).view(B*(K+1), C, H, W)
        
        features = img_cat
        for idx, layer in enumerate(self.backbone):
            # [B*(K+1),C,H,W]
            features = layer(features)

            # low-level features
            if idx == 0:
                features_0 = features.clone().detach()      

        _, fc, fh, fw = features.size()
        features = features.view(B, K+1, fc, fh, fw)
        sup_fts, qry_fts = features.split([K, 1], dim=1)            # [B, K, C, H, W] / [B, 1, C, H, W]
        feature_q = qry_fts.squeeze(1)              # [B, C, H, W]
        feature_s = sup_fts                         # [B, K, C, H, W]

        features_0 = self.proj_layer(features_0)
        # resize
        features_0 = F.interpolate(features_0, size=(fh, fw), mode='bilinear', align_corners=True)
        _, fc, fh, fw = features_0.size()
        features_0 = features_0.view(B, K+1, fc, fh, fw)                   # [B, K+1, c, h, w]
        sup_fts_0, qry_fts_0 = features_0.split([K, 1], dim=1)            # [B, K, c, h, w] / [B, 1, c, h, w]
        feature_q_0 = qry_fts_0.squeeze(1)              # [B, c, h, w]
        feature_s_0 = sup_fts_0                         # [B, K, c, h, w]

        return feature_q, feature_s, feature_q_0, feature_s_0




vit_configs = {
    'vit_depth': 10,                  # int, depth of the backbone
    'vit_stride': None,

    'drop_rate': 0.1,                 # float, drop rate used in the DropBlock of the purifier
    'block_size': 4,                  # int, block size used in the DropBlock of the purifier
    'drop_dim': 1,                    # int, 1 for 1D Dropout, 2 for 2D DropBlock
}

pretrained_dir = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))

pretrained_weights = {
    "ViT-B/8":          pretrained_dir / "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16":         pretrained_dir / "B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-384":     pretrained_dir / "B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    "ViT-B/16-i21k":    pretrained_dir / "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-i21k-384":pretrained_dir / "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    "ViT-S/16":         pretrained_dir / "S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "ViT-S/16-i21k":    pretrained_dir / "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "ViT-L/16":         pretrained_dir / "L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-L/16-384":     pretrained_dir / "L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",

    "DeiT-T/16":        pretrained_dir / "deit_tiny_distilled_patch16_224-b40b3cf7.pth",
    "DeiT-S/16":        pretrained_dir / "deit_small_distilled_patch16_224-649709d9.pth",
    "DeiT-B/16":        pretrained_dir / "deit_base_distilled_patch16_224-df68dfff.pth",
    "DeiT-B/16-384":    pretrained_dir / "deit_base_distilled_patch16_384-d0272ac0.pth",
}

model_urls = {
    "ViT-B/8":          "https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16":         "https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-384":     "https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    "ViT-B/16-i21k":    "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-i21k-384":"https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    "ViT-S/16":         "https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "ViT-S/16-i21k":    "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "ViT-L/16":         "https://storage.googleapis.com/vit_models/augreg/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-L/16-384":     "https://storage.googleapis.com/vit_models/augreg/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",

    "DeiT-T/16":        "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
    "DeiT-S/16":        "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
    "DeiT-B/16":        "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
    "DeiT-B/16-384":    "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",

}


class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = F.interpolate(x, (h * self.up, w * self.up), mode='bilinear', align_corners=True)
        x = x_up + self.layers(x)
        return x

# vit backbone
class Backbone_Vit(nn.Module):
    def __init__(self, args, backbone):
        super(Backbone_Vit, self).__init__()

        self.img_size = args.size
        self.shot = args.shot
        self.opt = vit_configs
        self.opt['shot'] = args.shot
        self.opt['backbone'] = args.backbone
        # dropout params
        self.drop_dim = self.opt['drop_dim']
        self.drop_rate = self.opt['drop_rate']
        self.drop2d_kwargs = {'drop_prob': self.opt['drop_rate'], 'block_size': self.opt['block_size']}
        
        # Check existence.
        pretrained = self.get_or_download_pretrained(backbone)
        
        self.backbone = vit.vit_model(backbone,
                                        self.img_size,
                                        pretrained=pretrained,
                                        num_classes=0,
                                        opt=self.opt,
                                        original=True)
        
        embed_dim = vit.vit_factory[backbone]['embed_dim']
        self.purifier = self.build_upsampler(embed_dim)

        # self.proj_layer = nn.Sequential(
        #     nn.Conv2d(768, 768, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True))

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))

    def forward(self, img_q, img_s):
        B, K, C, H, W = img_s.shape

        # [B，K+1, C, H, W]
        img_cat = torch.cat((img_s, img_q.view(B, 1, C, H, W)), dim=1).view(B*(K+1), C, H, W)
           
        inp = img_cat
        backbone_out = self.backbone(inp)
        features = backbone_out['out']
        # upsample
        features = self.purifier(features)

        _, fc, fh, fw = features.size()
        features = features.view(B, K+1, fc, fh, fw)
        sup_fts, qry_fts = features.split([K, 1], dim=1)            # [B, K, C, H, W] / [B, 1, C, H, W]
        feature_q = qry_fts.squeeze(1)              # [B, C, H, W]
        feature_s = sup_fts                         # [B, K, C, H, W]

        features_0 = backbone_out['out_0']
        # features_0 = self.proj_layer(features_0)
        # resize
        features_0 = F.interpolate(features_0, size=(fh, fw), mode='bilinear', align_corners=True)
        _, fc, fh, fw = features_0.size()
        features_0 = features_0.view(B, K+1, fc, fh, fw)                   # [B, K+1, c, h, w]
        sup_fts_0, qry_fts_0 = features_0.split([K, 1], dim=1)            # [B, K, c, h, w] / [B, 1, c, h, w]
        feature_q_0 = qry_fts_0.squeeze(1)              # [B, c, h, w]
        feature_s_0 = sup_fts_0                         # [B, K, c, h, w]

        return feature_q, feature_s, feature_q_0, feature_s_0


    @staticmethod
    def get_or_download_pretrained(backbone, progress=True):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')
        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file