#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

python experiments/compare_openness.py \
    --base_model tpn_slowonly \
    --baselines TPN_SlowOnly_Dropout_BALD TPN_SlowOnly_BNN_BALD TPN_SlowOnly_EDLlog_EDL TPN_SlowOnly_EDLlogAvUC_EDL TPN_SlowOnly_EDLlogNoKLAvUC_EDL \
    --thresholds 0.000096 [bnn?] [0.495783?] 0.495783 0.495800 \
    --ood_data HMDB \
    --ood_ncls 51 \
    --ind_ncls 101 \
    --result_png F1_openness_compare_HMDB.png
    

cd $pwd_dir
echo "Experiments finished!"