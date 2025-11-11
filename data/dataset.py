r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.fss import DatasetFSS
from data.deepglobe import DatasetDeepglobe
from data.isic import DatasetISIC
from data.lung import DatasetLung
from data import transformation as tf


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.train_transform = tf.Compose([tf.RandomResize(smin=0.9, smax=1.1),
                                          tf.RandomRotate(rotate=10, pad_type='reflect'),
                                          tf.RandomGaussianBlur(),
                                          tf.RandomHorizontallyFlip(),
                                          tf.RandomCrop(img_size, img_size, check=True, center=True, pad_type='reflect'),
                                          tf.ToTensor(mask_dtype='float'),
                                          tf.Normalize(cls.img_mean, cls.img_std)])
        cls.val_transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform = cls.train_transform if split == 'trn' else cls.val_transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
