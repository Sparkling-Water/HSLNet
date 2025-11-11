r""" Logging during training/testing """
import datetime
import logging
import os

import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        if self.benchmark == 'pascal':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 20
        elif self.benchmark == 'fss':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1000
        elif self.benchmark == 'deepglobe':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 6
        elif self.benchmark == 'isic':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 3
        elif self.benchmark == 'lung':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self):
        iou, fb_iou = self.compute_iou()
        
        msg = '***'
        msg += 'Testing mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        msg += '***'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, mode='train'):
        # output/backbone/*shot/exp*/test/{dataset}
        if mode != 'train':
            args.output_path = os.path.join(args.output_path, mode, f"{args.dataset}")
            cls.logpath = args.output_path
            os.makedirs(args.output_path, exist_ok=True)
        # output/backbone/*shot/exp*
        else:
            args.output_path = os.path.join(args.output_path, args.backbone, f"{args.shot}shot")
            os.makedirs(args.output_path, exist_ok=True)
            exp_list = sorted([int(d[3:]) for d in os.listdir(args.output_path) if d.startswith("exp")])
            if exp_list:
                cls.logpath = os.path.join(args.output_path, f"exp{exp_list[-1]+1}")
                args.output_path = cls.logpath
            else:
                cls.logpath = os.path.join(args.output_path, f"exp0")
                args.output_path = cls.logpath
            os.makedirs(cls.logpath)
        
        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Log arguments
        logging.info('\n:=========== Cross-Domain Few-shot Seg ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.module.state_dict(), os.path.join(cls.logpath, 'best_model.pth'))
        logging.info('***Model saved @%d w/ val. mIoU: %5.2f   ***' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        Logger.info('Learnable # params: %.2fM' % (learnable_params/1e6))
        Logger.info('Total # params: %.2fM' % (total_params/1e6))