#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch --nproc_per_node=4 --use_env \
# training/train.py --distributed --port 12364 \
# --pc_normalize \
# --model2d_tile resnet_safa --model2d_pano resnet_safa \
# --model3d dgcnn --feat_dim 512 \
# --loss MultiInfoNCELoss --margin 0.07 \
# --npoints 1024 --nneighbor 20  --fuse 2to3 \
# --lr 1e-4 --optimizer SAM --wd 0.03 --epochs 60 --scheduler CosineAnnealingLR

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --use_env \
training/train.py --distributed --port 12364 \
--feat_dim 4096 \
--pre_model_name 'resnetsafa_asam_simple' \ 
--margin 0.07 --lr 1e-4 \
--optimizer SAM --wd 0.03 --epochs 60 \
--scheduler 'CosineAnnealingLR' \
--num_layers 6 --num_heads 8 --seq_len 5