#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

source activate mmaction

#  get the BALD threshold for I3D_Dropout model trained on UCF-101
CUDA_VISIBLE_DEVICES=$1 python experiments/get_threshold.py \
    --config configs/recognition/i3d/finetune_ucf101_i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
    --checkpoint work_dirs/finetune_ucf101_i3d_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
    --train_data data/ucf101/ucf101_train_split_1_videos.txt \
    --batch_size 8 \
    --uncertainty BALD


#  get the BALD threshold for I3D_BNN model trained on UCF-101
CUDA_VISIBLE_DEVICES=$1 python experiments/get_threshold.py \
    --config configs/recognition/i3d/finetune_ucf101_i3d_bnn_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
    --checkpoint work_dirs/finetune_ucf101_i3d_bnn_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
    --train_data data/ucf101/ucf101_train_split_1_videos.txt \
    --batch_size 8 \
    --uncertainty BALD