#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 \
# python eval/feature_extract.py --pc_normalize \
# --model3d none \
# --model2d_pano resnet_safa \
# --model2d_tile resnet_safa \
# --feat_dim 4096 \
# --weights 'weights/resnetsafa_asam_simple.pth' \
# --npoints 1024 --nneighbor 20 \
# --fuse 'concat' \
# --eval_files 'trainstreetlearnU_cmu5kU' \
# --exp_name 'none' \
# --pca_dim 128

CUDA_VISIBLE_DEVICES=0 \
python eval/evaluate_seq.py \
--feat_dim 4096 \
--pre_model_name 'resnetsafa_dgcnn_asam_2to3_up' \
--eval_files 'hudsonriver5kU' \
--exp_name 'none' \
--pca_dim 128
