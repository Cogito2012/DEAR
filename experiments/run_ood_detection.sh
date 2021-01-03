#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

source activate mmaction

# I3D model with Dropout trained on UCF-101, testing on HMDB-51 val split_1 set
CUDA_VISIBLE_DEVICES=$1 python experiments/ood_detection.py \
    --config configs/recognition/i3d/finetune_ucf101_i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
    --checkpoint work_dirs/finetune_ucf101_i3d_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
    --ind_data data/ucf101/ucf101_val_split_1_videos.txt \
    --ood_data data/hmdb51/hmdb51_val_split_1_videos.txt \
    --uncertainty BALD \
    --result_tag I3D_Dropout_BALD


# # BNN model trained on UCF-101, testing on HMDB-51 val split_1 set
# CUDA_VISIBLE_DEVICES=$1 python experiments/ood_detection.py \
#     --config configs/recognition/i3d/finetune_ucf101_i3d_bnn_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
#     --checkpoint work_dirs/finetune_ucf101_i3d_bnn_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
#     --ind_data data/ucf101/ucf101_val_split_1_videos.txt \
#     --ood_data data/hmdb51/hmdb51_val_split_1_videos.txt \
#     --uncertainty BALD \
#     --result_tag I3D_BNN_BALD


# # EDL model trained on UCF-101, testing on HMDB-51 val split_1 set
# CUDA_VISIBLE_DEVICES=$1 python experiments/ood_detection_edl.py \
#     --config configs/recognition/i3d/finetune_ucf101_i3d_edllogloss_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
#     --checkpoint work_dirs/finetune_ucf101_i3d_edllogloss_r50_dense_32x2x1_100e_kinetics400_rgb/latest.pth \
#     --label_names data/ucf101/annotations/classInd.txt \
#     --ind_data data/ucf101/ucf101_val_split_1_videos.txt \
#     --ood_data data/hmdb51/hmdb51_val_split_1_videos.txt \
#     --result_tag I3D_EDLlog_EDL