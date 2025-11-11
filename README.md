# Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation
Official code for AAAI 2026 paper: Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation

## Datasets
You can follow [PATNet](https://github.com/slei109/PATNet) to prepare the source domain and target domain datasets.

### Source domain: 

* **PASCAL VOC2012**:

    Download PASCAL VOC2012 devkit (train/val data):
    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    ```
    Download PASCAL VOC2012 SDS extended mask annotations from [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].

### Target domains: 

* **Deepglobe**:

    Home: http://deepglobe.org/

    Direct: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
    
    Preprocessed Data: https://drive.google.com/file/d/10qsi1NRyFKFyoIq1gAKDab6xkbE0Vc74/view?usp=sharing

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018
    
    Class Information: data/isic/class_id.csv

* **Chest X-ray**:

    Home: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

    Direct: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

* **FSS-1000**:

    Home: https://github.com/HKUSTCV/FSS-1000

    Direct: https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view


## Run the code
### Generating Spixel Masks
> ```bash
> python3 generate_spixel.py --dataset lung
> ```

### Training
> ```bash
> CUDA_VISIBLE_DEVICES=0 python -W ignore train.py --datapath ./datasets --backbone vit_b --size 480 --batch-size 8 --episode 6000 --shot 1 --perturb
> ```

### Testing
> ```bash
> CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset lung --datapath ./datasets --backbone vit_b --size 480 --checkpoint output/vit_b/1shot/exp0/best_model.pth --batch-size 1 --shot 1
> ```


## Acknowledgement
Our code is built upon the foundations of [SSP](https://github.com/fanq15/SSP), [PATNet](https://github.com/slei109/PATNet), [CDSpixel](https://github.com/rookiie/CDSpixel), [ABCDFSS](https://github.com/Vision-Kek/ABCDFSS) and [GPRN](https://github.com/CVL-hub/GPRN), we appreciate the authors for their excellent contributions!
