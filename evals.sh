#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 \
# python eval/feature_extract.py --pc_normalize \
# --model3d none \
# --model2d_pano resnet_safa \
# --model2d_tile resnet_safa \
# --feat_dim 512 \
# --weights 'weights/resnetsafa_asam_simple.pth' \
# --npoints 1024 --nneighbor 20 \
# --fuse 'concat' \
# --eval_files 'trainstreetlearnU_cmu5kU' \
# --exp_name 'none' \
# --pca_dim 128

# CUDA_VISIBLE_DEVICES=0 \
# python eval/evaluate_seq.py \
# --feat_dim 4096 --share \
# --pre_model_name 'resnetsafa_asam_simple' \
# --eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
# --exp_name 'none' \
# --num_layers 1 --num_heads 8 --seq_len 5 \
# --pca_dim 80 --model_type 'transmixer' \
# --weights 'weights3090/model_20231026_2250_best_top1.pth'

CUDA_VISIBLE_DEVICES=0 \
python eval/evaluate_seq_v2.py \
--feat_dim 4096 --share \
--pretrained 'weights/resnetsafa_asam_simple.pth' \
--eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
--exp_name 'none' \
--num_layers 1 --num_heads 8 --seq_len 5 \
--pca_dim 80 --model_type 'transmixer' \
--map_type 'single' 