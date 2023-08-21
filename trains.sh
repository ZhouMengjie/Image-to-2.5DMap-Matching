#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --use_env \
training_pickle/train.py --distributed --port 12364 \
--pc_normalize \
--model2d_tile resnet_safa --model2d_pano resnet_safa \
--model3d pt2 --feat_dim 4096 \
--loss MultiInfoNCELoss --margin 0.07 \
--npoints 1024 --nneighbor 20  --fuse add --fc fc \
--lr 1e-4 --optimizer SAM --wd 0.03 --epochs 60 --scheduler CosineAnnealingLR \
# --weights '/cpfs01/shared/MMG/MMG_hdd/zhoumengjie/weights/model_MinkLocMultimodal_20230718_1713_latest.pth' \
# --epoch 50