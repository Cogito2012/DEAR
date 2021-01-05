import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

def parse_args():
    '''Command instruction:
        source activate mmaction
        CUDA_VISIBLE_DEVICES=0 python experiments/compare_openness.py \
            --baselines I3D_Dropout_BALD I3D_BNN_BALD \
            --styles '-b' '-r' \
            --ind_ncls 101 \
            --ood_ncls 51
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--baselines', nargs='+', default=['I3D_Dropout_BALD', 'I3D_BNN_BALD'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.048563, 0.006867])
    parser.add_argument('--styles', nargs='+', default=['-b', '-r'])
    parser.add_argument('--ind_ncls', type=int, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    args = parser.parse_args()
    return args


def main():

    plt.figure(figsize=(5,3))  # (w, h)
    for style, thresh, baseline in zip(args.styles, args.thresholds, args.baselines):
        result_file = os.path.join('./experiments/results', baseline + '_result.npz')
        assert os.path.exists(result_file), "File not found! Run ood_detection first!"
        # load the testing results
        results = np.load(result_file, allow_pickle=True)
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
        ind_results = results['ind_pred']  # (N1,)
        ood_results = results['ood_pred']  # (N2,)
        ind_labels = results['ind_label']
        ood_labels = results['ood_label']

        # close-set accuracy (multi-class)
        acc = accuracy_score(ind_labels, ind_results)
        # open-set auc-roc (binary class)
        preds = np.concatenate((ind_results, ood_results), axis=0)
        uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        preds[uncertains > thresh] = 1
        preds[uncertains <= thresh] = 0
        labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
        aupr = roc_auc_score(labels, preds)
        print('Model: %s, ClosedSet Accuracy (multi-class): %.3lf, OpenSet AUC (bin-class): %.3lf'%(baseline, acc * 100, aupr))
        
        # open set F1 score (multi-class)
        ind_results[ind_uncertainties > thresh] = args.ind_ncls  # falsely rejection
        macro_F1_list = [f1_score(ind_labels, ind_results, average='macro')]
        openness_list = [0]
        for n in range(args.ood_ncls):
            ncls_novel = n + 1
            openness = (1 - np.sqrt((2 * args.ind_ncls) / (2 * args.ind_ncls + ncls_novel))) * 100
            openness_list.append(openness)
            # select the subset of ood samples 
            ood_sub_results = ood_results[ood_labels <= n]
            ood_sub_uncertainties = ood_uncertainties[ood_labels <= n]
            ood_sub_results[ood_sub_uncertainties > thresh] = args.ind_ncls  # correctly rejection
            ood_sub_labels = np.ones_like(ood_sub_results) * args.ind_ncls
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1 = f1_score(labels, preds, average='macro')
            macro_F1_list.append(macro_F1)

        # draw comparison curves
        plt.plot(openness_list, macro_F1_list, style, linewidth=2)

    plt.xlim(0, 11)
    plt.ylim(0.7, 1.01)
    plt.xlabel('Openness (%)')
    plt.ylabel('macro F1')
    plt.grid('on')
    plt.legend(args.baselines)
    plt.tight_layout()
    plt.savefig('./experiments/results/F1_openness_compare.png')


if __name__ == "__main__":

    args = parse_args()

    main()