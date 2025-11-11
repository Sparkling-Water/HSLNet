import math
import torch
from torch import nn
import torch.nn.functional as F

from model.backbone import Backbone_Res, Backbone_Vit
from model.base.model_utils import DoublePerturb, SelfAttention
from common.utils import calcthresh


# HSLNet
class HSLNet(nn.Module):
    def __init__(self, args):
        super(HSLNet, self).__init__()
        self.perturb = args.perturb
        self.shot= args.shot
        self.gray = True if args.dataset in ['lung'] else False

        # CNN
        if args.backbone == 'resnet50':
            self.random_std = 0.1
            self.feat_dim_list = [3, 128, 256, 512, 1024]
            # ResNet
            self.backbone = Backbone_Res(args.backbone)
        # ViT
        elif args.backbone == 'vit_b':
            self.random_std = 0.6
            self.feat_dim_list = [3, 768]
            # ViT-base
            self.backbone = Backbone_Vit(args, 'ViT-B/16-384')
        else:
            pass

        self.feat_dim = self.feat_dim_list[-1]
        self.perturb_layer = DoublePerturb(self.feat_dim_list[0], random_size=3, random_std=self.random_std, prob=1.0, mask_size=9) if self.perturb else None
        self.attn_layer = nn.ModuleList([SelfAttention(self.feat_dim, nhead=4, dropout=0.0, activation='relu'),
                                         SelfAttention(self.feat_dim, nhead=4, dropout=0.0, activation='relu')])

        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, img_q, mask_q, img_s_list, mask_s_list, spixel_mask_q, spixel_mask_s_list):
        B, C, H, W = img_q.shape
        K = len(img_s_list)
        L, SH, SW = spixel_mask_q.shape[1:]

        # Dual Style Randomization (DSR)
        if self.perturb and self.training:
            img_q = self.perturb_layer(img_q, mask_q, spixel_mask_q)
            img_s_list = [self.perturb_layer(feat, mask_s_list[i], spixel_mask_s_list[i]) for i, feat in enumerate(img_s_list)]
        
        # [B,K,C,H,W]
        img_s = torch.stack(img_s_list, dim=0).permute(1,0,2,3,4).contiguous()

        # extract features
        feature_q, feature_s, feature_q_0, feature_s_0 = self.backbone(img_q, img_s)
        FC, FH, FW = feature_q.shape[1:]
        # for grayscale images
        if self.gray:
            feature_q = F.instance_norm(feature_q)
            feature_s = F.instance_norm(feature_s.contiguous().view(B*K, FC, FH, FW)).view(B, K, FC, FH, FW)
            feature_q_0 = F.instance_norm(feature_q_0)
            feature_s_0 = F.instance_norm(feature_s_0.contiguous().view(B*K, FC, FH, FW)).view(B, K, FC, FH, FW)

        # query mask, [B,FH,FW]
        mask_q = F.interpolate(mask_q.unsqueeze(1), size=(FH, FW), mode='nearest').squeeze(1)
        # support mask, [B,K,FH,FW]
        mask_s = torch.stack(mask_s_list, dim=0).permute(1,0,2,3).contiguous()
        mask_s = F.interpolate(mask_s, size=(FH, FW), mode="nearest")
        # query spixel mask, [B,L,FH,FW]
        spixel_mask_q = F.interpolate(spixel_mask_q.float(), size=(FH, FW), mode='nearest')
        spixel_mask_q = spixel_mask_q
        # support spixel mask, [B,K,L,FH,FW]
        spixel_mask_s = torch.stack(spixel_mask_s_list, dim=0).permute(1,0,2,3,4).contiguous()
        spixel_mask_s = F.interpolate(spixel_mask_s.contiguous().view(B*K, L, SH, SW).float(), size=(FH, FW), mode="nearest").view(B, K, L, FH, FW)

        output = {}
        if self.training:
            # foreground(target class) and background prototypes pooled from K support features
            feature_fg_list = []    # [K,B,C]
            feature_bg_list = []    # [K,B,C]
            supp_out_ls = []        # [K,B,2,H,W]
            for k in range(len(img_s_list)):
                feature_fg = self.masked_average_pooling(feature_s[:, k, :, :, :],
                                                        (mask_s[:, k, :, :] == 1).float())[None, :]
                feature_bg = self.masked_average_pooling(feature_s[:, k, :, :, :],
                                                        (mask_s[:, k, :, :] == 0).float())[None, :]
                feature_fg_list.append(feature_fg)
                feature_bg_list.append(feature_bg)

                supp_similarity_fg = F.cosine_similarity(feature_s[:, k, :, :, :], feature_fg.squeeze(0)[..., None, None],
                                                        dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s[:, k, :, :, :], feature_bg.squeeze(0)[..., None, None],
                                                        dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0
                # [B,2,H,W]
                supp_out = F.interpolate(supp_out, size=(H, W), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

            # average K foreground prototypes and K background prototypes, [B,C,1,1]
            FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
            BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

            # measure the similarity of query features to fg/bg prototypes
            out_0 = self.similarity_func(feature_q, FP, BP)
            SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)

            # [B,C,1,1]
            FP_1 = FP * 0.5 + SSFP_1 * 0.5
            # [B,C,H,W]
            BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

            # [B,2,H,W]
            out_1 = self.similarity_func(feature_q, FP_1, BP_1)
            out_1 = F.interpolate(out_1, size=(H, W), mode="bilinear", align_corners=True)
            output['out'] = out_1

            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)
            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(H, W), mode="bilinear", align_corners=True)
            # [B,2,H,W]
            output['self_out'] = self_out
            output['supp_out'] = supp_out_ls

        # Hierarchical Semantic Mining (HSM)
        # multi-scale region features
        region_features_list = self.get_region_features(feature_q, feature_q_0, spixel_mask_q, feature_s, feature_s_0, spixel_mask_s)

        # fusion region features
        region_feat_s = torch.zeros_like(feature_s).to(feature_s.device)
        region_feat_q = torch.zeros_like(feature_q).to(feature_q.device)
        for i in range(L-1):
            # [B,K+1,C,H,W]
            region_features = region_features_list[i]
            # split
            region_feature_s, region_feature_q = region_features.split([K, 1], dim=1)

            # [B,K,C,H,W]
            region_feat_s += (region_feature_s)
            # [B,C,H,W]
            region_feat_q += (region_feature_q.squeeze(1))
        
        # enhanced features
        en_feature_q = self.relu((feature_q + region_feat_q)/(L+1))
        en_feature_s = self.relu((feature_s + region_feat_s)/(L+1))
        
        # foreground(target class) and background prototypes pooled from K support features
        proto_fg_list = []    # [K,B,C]
        proto_bg_list = []    # [K,B,C]
        en_supp_out_ls = []  # [K,B,2,H,W]
        for k in range(len(img_s_list)):
            # [1,B,C]
            proto_fg = self.masked_average_pooling(en_feature_s[:, k, :, :, :],
                                                    (mask_s[:, k, :, :] == 1).float())[None, :]
            proto_bg = self.masked_average_pooling(en_feature_s[:, k, :, :, :],
                                                    (mask_s[:, k, :, :] == 0).float())[None, :]
            proto_fg_list.append(proto_fg)
            proto_bg_list.append(proto_bg)

        # [B,C,1,1]
        EN_FP = torch.mean(torch.cat(proto_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        EN_BP = torch.mean(torch.cat(proto_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # measure the similarity of query features to fg/bg prototypes
        en_out_0 = self.similarity_func(en_feature_q, EN_FP, EN_BP)
        EN_SSFP_1, EN_SSBP_1, EN_ASFP_1, EN_ASBP_1 = self.SSP_func(en_feature_q, en_out_0)

        # [B,C,1,1]
        EN_FP_1 = EN_FP * 0.5 + EN_SSFP_1 * 0.5
        # [B,C,H,W]
        EN_BP_1 = EN_SSBP_1 * 0.3 + EN_ASBP_1 * 0.7

        # similarity map, [B,2,H,W]
        en_out_1 = self.similarity_func(en_feature_q, EN_FP_1, EN_BP_1)
        en_out_1 = F.interpolate(en_out_1, size=(H, W), mode="bilinear", align_corners=True)
        output['en_out'] = en_out_1

        # Prototype Confidence-modulated Thresholding (PCMT)
        pred = output['en_out']
        # foreground confidence map
        pred_conf = ((pred[:, 1, :, :] - pred[:, 0, :, :]) / 10.0).detach()

        # prototype confidence
        sim_intra = F.cosine_similarity(EN_SSFP_1.squeeze(-1).squeeze(-1), EN_FP.squeeze(-1).squeeze(-1), dim=1)
        sim_inter1 = F.cosine_similarity(EN_SSFP_1.squeeze(-1).squeeze(-1), EN_BP.squeeze(-1).squeeze(-1), dim=1)
        sim_inter2 = F.cosine_similarity(EN_FP.squeeze(-1).squeeze(-1), EN_BP_1.mean(dim=(2,3)), dim=1)
        sim_diff = (sim_intra - 0.5 * (sim_inter1 + sim_inter2))
        rate = 1.0 / (1.0 + torch.exp(40 * (sim_diff - 0.1)))
        # thresh = rate * OTSU
        thresh = calcthresh(pred_conf, method='otsus') * rate

        pred_mask = (pred_conf > thresh.unsqueeze(-1).unsqueeze(-1)).float()

        return pred_mask, output


    def SSP_func(self, feature_q, out):
        bs, ch = feature_q.shape[:2]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7  # 0.9 #0.6
            bg_thres = 0.6  # 0.6
            cur_feat = feature_q[epi].view(ch, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            # [1,C]
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)

            cur_feat_norm_t = cur_feat_norm.t()  # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0
            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
            # [1,C,H,W]
            fg_proto_local = fg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0)
            bg_proto_local = bg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0)

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto, [B,C,1,1]
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto, [B,C,H,W]
        new_fg_local = torch.cat(fg_local_ls, 0)
        # [B,C,H,W]
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local
    

    def get_region_features(self, feature_q, feature_q_0, spixel_mask_q, feature_s, feature_s_0, spixel_mask_s):
        B, K, C, H, W = feature_s.shape
        L, h, w = spixel_mask_q.shape[1:]

        # [B*(K+1), C, H, W]
        feature_cat = torch.cat((feature_s, feature_q.unsqueeze(1)), dim=1).contiguous().view(B*(K+1), C, H, W)
        feature_0_cat = torch.cat((feature_s_0, feature_q_0.unsqueeze(1)), dim=1).contiguous().view(B*(K+1), C, H, W)
        # [B*(K+1), L, H, W]
        spixel_mask_cat = torch.cat((spixel_mask_s, spixel_mask_q.unsqueeze(1)), dim=1).contiguous().view(B*(K+1), L, h, w)

        region_features_list = []
        for i in range(L):
            # spixel mask, [B*(K+1),N,H,W]
            spixel_mask = self.spixel_mask_process(spixel_mask_cat[:,i,:,:], H, W)
            N = spixel_mask.shape[1]
            # region proto, [B*(K+1),N,C]
            spixel_feat_0 = self.multi_masked_average_pooling(feature_0_cat, spixel_mask)
            spixel_feat = self.multi_masked_average_pooling(feature_cat, spixel_mask)

            spixel_feat_0 = self.attn_layer[0](spixel_feat_0.permute(1,0,2).contiguous())
            spixel_feat_0 = self.attn_layer[1](spixel_feat_0).permute(1,0,2).contiguous()

            enhance_spixel_feat = 0.8 * spixel_feat + 0.2 * spixel_feat_0

            # RMAP, [B*(K+1),C,H,W]
            enhance_feature = (enhance_spixel_feat.permute(0, 2, 1).contiguous() @ spixel_mask.contiguous().view(B*(K+1), N, -1)).view(B, (K+1), C, H, W)
            region_features_list.append(enhance_feature)

        return region_features_list
    
    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        # [B,2,H,W]
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out


    # MAP，[B,C,H,W]-->[B,C]
    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    
    def spixel_mask_process(self, spixel_mask, H, W):
        B, spixelH, spixelW = spixel_mask.shape
        N = spixel_mask.max().long()
        # 0-(N-1) is mask_id， N is background
        index_mask = spixel_mask.long()-1
        # one-hot,[B,N,H,W]
        target_masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :N].permute((0, 3, 1, 2)).float()
        return target_masks
    
    def multi_masked_average_pooling(self, feature, mask):
        B, C, W, H = feature.shape
        _, N, _, _ = mask.shape

        # [B,N,H*W]
        _mask = mask.view(B, N, -1)
        # [B,H*W,C]
        _feature = feature.view(B, C, -1).permute(0, 2, 1).contiguous()
        # [B,N,C]
        feature_sum = _mask @ _feature # B x N x C
        masked_sum = torch.sum(_mask, dim=2, keepdim=True) # B x N x 1
        masked_average_pooling = torch.div(feature_sum, masked_sum + 1e-5)
        return masked_average_pooling
    
    def compute_loss(self, out_ls, mask_q, mask_s_list):
        mask_s = torch.cat(mask_s_list, dim=0)
        mask_s = mask_s.long()
        mask_q = mask_q.long()

        loss = self.criterion(out_ls['out'], mask_q) + self.criterion(out_ls['self_out'], mask_q) + self.criterion(torch.cat(out_ls['supp_out'], dim=0), mask_s) * 0.2
        loss += self.criterion(out_ls['en_out'], mask_q)
        
        return loss