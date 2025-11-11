import random
import math
import numpy as np
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RandomConv(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, std=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.std = std
        self.padding = kernel_size // 2
        
    def forward(self, x):
        # N(0, std^2)
        weight = (torch.randn(
            self.in_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            device=x.device
        ) * self.std)

        x = F.conv2d(x, weight, padding=self.padding, bias=None)
        return x
    

class MaskDilator(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def forward(self, mask):      
        # [B, 1, H, W]
        new_mask = mask.float().unsqueeze(1)
        
        # maxpooling
        new_mask = F.max_pool2d(
            new_mask,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1
        )
        # avgpooling
        new_mask = F.avg_pool2d(
            new_mask,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1
        )
        
        # [B, H, W]
        new_mask = new_mask.squeeze(1)
        return new_mask
    

# Dual Style Randomization
class DoublePerturb(nn.Module):
    def __init__(self, feat_dim=3, random_size=3, random_std=0.1, prob=1.0, mask_size=9):
        super(DoublePerturb, self).__init__()
        self.random_conv = RandomConv(feat_dim, random_size, random_std)
        self.mask_dilator = MaskDilator(kernel_size=mask_size)
        
        self.feat_dim = feat_dim
        self.prob = prob

    def forward(self, feature, mask, spixel_mask):
        # foreground style randomization
        if random.random() < self.prob:
            feature = self.foreground_perturb(feature, mask, spixel_mask)
        
        # global style randomization
        if random.random() < self.prob:
            feature = self.global_perturb(feature)

        return feature

    def foreground_perturb(self, feature, mask, spixel_mask):
        bs, nc, fh, fw = feature.shape
        # [B,H,W]
        mask = F.interpolate(mask.unsqueeze(1).float(), size=feature.shape[-2:], mode='nearest').squeeze(1)
        # preprocess
        dilate_mask = self.mask_dilator(mask)
        # [B,L,H,W]
        spixel_mask = F.interpolate(spixel_mask.float(), size=feature.shape[-2:], mode='nearest')
        
        perturb_feature_list = []
        for epi in range(bs):
            # [C,H,W]
            cur_feat = feature[epi]
            # [H,W]
            cur_dilate_mask = dilate_mask[epi]
            cur_spixel_mask = spixel_mask[epi][0]

            # region id
            unique_labels = torch.unique(cur_spixel_mask)
            # random select
            selected_label = np.random.choice(unique_labels.cpu().numpy())
            # region mask
            selected_mask = torch.zeros_like(cur_spixel_mask)
            selected_mask[cur_spixel_mask == selected_label] = 1.0

            # if selected region too small, skip perturbation
            M = selected_mask.sum()
            if M < (fh*fw)/unique_labels.shape[0] or cur_dilate_mask.sum() < (fh*fw)/unique_labels.shape[0]:
                perturb_feature_list.append(cur_feat.view(1, nc, fh, fw))
                continue

            # bounding box
            coords1 = torch.nonzero(cur_dilate_mask)
            coords2 = torch.nonzero(selected_mask)
            y1_min, x1_min = torch.min(coords1, dim=0).values
            y1_max, x1_max = torch.max(coords1, dim=0).values
            y2_min, x2_min = torch.min(coords2, dim=0).values
            y2_max, x2_max = torch.max(coords2, dim=0).values
            
            # crop patches
            patch1 = cur_feat[:, y1_min:y1_max+1, x1_min:x1_max+1]  # [C, H1, W1]
            patch2 = cur_feat[:, y2_min:y2_max+1, x2_min:x2_max+1]  # [C, H2, W2]
            patch2 = F.interpolate(patch2.unsqueeze(0), size=patch1.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)

            # sample perturbation weight
            perturb_weight = (0.25 * torch.randn_like(patch1.mean(dim=(1, 2), keepdim=True))).unsqueeze(0).clamp(-1.0, 1.0)

            # foreground phase
            feature_fg_freq = torch.fft.fftshift(torch.fft.fft2(patch1.unsqueeze(0)))
            phase = torch.angle(feature_fg_freq)
            # local amplitude
            feature_bg_freq = torch.fft.fftshift(torch.fft.fft2(patch2.unsqueeze(0)))
            bg_amplitude = torch.abs(feature_bg_freq)
            # fusion amplitude
            amplitude = perturb_weight * bg_amplitude + (1 - perturb_weight) * torch.abs(feature_fg_freq)

            fusion_freq = torch.polar(amplitude, phase)
            patch1 = torch.fft.ifft2(torch.fft.ifftshift(fusion_freq)).real

            perturb_feat = cur_feat.clone()
            perturb_feat[:, y1_min:y1_max+1, x1_min:x1_max+1] = patch1
            perturb_feat = perturb_feat * cur_dilate_mask.unsqueeze(0) + cur_feat * (1 - cur_dilate_mask.unsqueeze(0))

            perturb_feature_list.append(perturb_feat.view(1, nc, fh, fw))

        # [B,C,H,W]
        perturb_feature = torch.cat(perturb_feature_list, dim=0)

        return perturb_feature

    def global_perturb(self, feature):
        random_feature = self.random_conv(feature)

        # phase
        feature_freq = torch.fft.fftshift(torch.fft.fft2(feature))
        phase = torch.angle(feature_freq)
        # random amplitude
        random_feature_freq = torch.fft.fftshift(torch.fft.fft2(random_feature))
        amplitude = torch.abs(random_feature_freq)

        fusion_freq = torch.polar(amplitude, phase)
        perturb_feature = torch.fft.ifft2(torch.fft.ifftshift(fusion_freq)).real

        return perturb_feature


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None):
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask)