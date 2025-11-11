## 1-shot ##
# train
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py --datapath ./datasets --backbone vit_b --size 480 --batch-size 8 --episode 6000 --shot 1 --perturb
# test
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --dataset lung --datapath ./datasets --backbone vit_b --size 480 --checkpoint output/vit_b/1shot/exp0/best_model.pth --batch-size 1 --shot 1

## 5-shot ##
# train
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py --datapath ./datasets --backbone vit_b --size 480 --batch-size 6 --episode 6000 --shot 5 --perturb
# test
CUDA_VISIBLE_DEVICES=0,1 python -W ignore test.py --dataset lung --datapath ./datasets --backbone vit_b --size 480 --checkpoint ./output/vit_b/5shot/exp0/best_model.pth --batch-size 1 --shot 5
