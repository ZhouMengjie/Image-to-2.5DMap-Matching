#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 \
# python eval/evaluate_pickle.py --pc_normalize \
# --model3d dgcnn \
# --model2d_pano resnet_safa \
# --model2d_tile resnet_safa \
# --feat_dim 512 \
# --weights 'weights/resnetsafa_dgcnn_asam_2to3_up.pth' \
# --npoints 1024 --nneighbor 20 \
# --fuse '2to3' \
# --eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
# --exp_name 'none' \
# --pca_dim 128

# CUDA_VISIBLE_DEVICES=0 \
# python eval/feature_extract.py --pc_normalize \
# --model3d pointnet \
# --model2d_pano resnet_safa \
# --model2d_tile resnet_safa \
# --feat_dim 512 \
# --weights 'weights/resnetsafa_pointnet_asam_2to3_up.pth' \
# --npoints 1024 --nneighbor 20 \
# --fuse '2to3' \
# --eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
# --exp_name 'none' \
# --pca_dim 16

CUDA_VISIBLE_DEVICES=0 \
python eval/evaluate_seq.py \
--feat_dim 4096 --share \
--pre_model_name 'resnetsafa_asam_simple_16' \
--eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
--exp_name 'none' \
--num_layers 1 --num_heads 8 --seq_len 5 \
--pca_dim 80 --model_type 'baseline' \
--pool 'avg_pool' \
# --weights 'weights3090/model_20231026_2215_best_top1.pth'

# CUDA_VISIBLE_DEVICES=0 \
# python eval/evaluate_seq_v2.py \
# --feat_dim 4096 --share \
# --pretrained 'weights/resnetsafa_dgcnn_asam_2to3_up.pth' \
# --eval_files 'trainstreetlearnU_cmu5kU' \
# --exp_name 'none' \
# --num_layers 1 --num_heads 8 --seq_len 5 \
# --pca_dim 80 --model_type 'transmixer' \
# --map_type 'multi' --use_cloud \
# --weights 'weights3090/model_20231030_1613_best_top1.pth'

# CUDA_VISIBLE_DEVICES=0 \
# python eval/evaluate_video.py \
# --feat_dim 4096 \
# --eval_files 'trainstreetlearnU_cmu5kU' \
# --exp_name 'none' \
# --seq_len 5 \
# --pca_dim 80 \
# --weights 'weights3090/model_20231101_2045_best_top1.pth'