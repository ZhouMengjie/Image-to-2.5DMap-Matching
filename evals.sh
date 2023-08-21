#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python eval/evaluate_pickle.py --pc_normalize \
--model3d dgcnn --use_feat \
--model2d_pano resnet_safa \
--model2d_tile none \
--feat_dim 4096 \
--weights 'weights3090/model_MinkLocMultimodal_20230815_0458_best_top1.pth' \
--npoints 1024 --nneighbor 20 \
--fuse 'concat' \
--eval_files 'hudsonriver5kU,wallstreet5kU,unionsquare5kU' \
--exp_name 'none' \
--pca_dim 128
