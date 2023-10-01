#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python eval/feature_extract.py --pc_normalize \
--model3d dgcnn \
--model2d_pano resnet_safa \
--model2d_tile resnet_safa \
--feat_dim 512 \
--weights 'weights/resnetsafa_dgcnn_asam_2to3_up.pth' \
--npoints 1024 --nneighbor 20 \
--fuse '2to3' \
--eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
--exp_name 'none' \
--pca_dim 128
