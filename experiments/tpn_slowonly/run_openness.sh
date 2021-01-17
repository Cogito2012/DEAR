#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
case ${OOD_DATA} in
    HMDB)
    NUM_CLASSES=51
    ;;
    MiT)
    NUM_CLASSES=305
    ;;
    *)
    echo "Invalid OOD Dataset: "${OOD_DATA}
    exit
    ;;
esac

# OOD Detection comparison
python experiments/compare_openness.py \
    --base_model tpn_slowonly \
    --baselines TPN_SlowOnly_Dropout_BALD TPN_SlowOnly_BNN_BALD TPN_SlowOnly_EDLlog_EDL TPN_SlowOnly_EDLlogAvUC_EDL TPN_SlowOnly_EDLlogNoKLAvUC_EDL \
    --thresholds 0.000096 0.000007 0.495784 0.495783 0.495800 \
    --ood_data ${OOD_DATA} \
    --ood_ncls ${NUM_CLASSES} \
    --ind_ncls 101 \
    --result_png F1_openness_compare_${OOD_DATA}.png
    

cd $pwd_dir
echo "Experiments finished!"