#!/usr/bin/env bash
# export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO

# python -m torch.distributed.launch --use_env \
# --nproc_per_node=${KUBERNETES_CONTAINER_RESOURCE_GPU} \
# --master_addr=${MASTER_ADDR} \
# --master_port=${MASTER_PORT} \
# --nnodes=${WORLD_SIZE} \
# --node_rank=${RANK} \
# training_pickle/train.py --distributed --port 12367 \
# --pc_normalize --batch_size 16 --val_batch_size 10 \
# --model2d_tile none --model2d_pano resnet_safa \
# --model3d dgcnn --feat_dim 4096 \
# --loss MultiInfoNCELoss --margin 0.07 \
# --npoints 1024 --nneighbor 20  --fuse concat \
# --lr 1e-4 --optimizer SAM --wd 0.03 --epochs 60 --scheduler CosineAnnealingLR

python -m torch.distributed.launch --use_env \
--nproc_per_node=${KUBERNETES_CONTAINER_RESOURCE_GPU} \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
--nnodes=${WORLD_SIZE} \
--node_rank=${RANK} \
training_seq/train.py --distributed --port 12364 \
--feat_dim 4096 --batch_size 24 \
--pre_model_name 'resnetsafa_asam_simple' \
--margin 0.07 --lr 1e-4 \
--optimizer SAM --wd 0.03 --epochs 60 \
--scheduler 'CosineAnnealingLR' \
--num_layers 3 --num_heads 8 --seq_len 5 \
--model_type 'seqnet'


