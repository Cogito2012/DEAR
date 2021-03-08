import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# from mmaction.core.evaluation import confusion_matrix

def confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                      ood_labels, ood_results, ood_uncertainties,
                      threshold, know_ood_labels=False, normalize=True):
    num_indcls = max(ind_labels) + 1
    num_oodcls = max(ood_labels) + 1
    confmat = np.zeros((num_indcls + num_oodcls, num_indcls + num_oodcls), dtype=np.float32)
    for rlabel, plabel, uncertain in zip(ind_labels, ind_results, ind_uncertainties):
        if uncertain > threshold:
            # known --> unknown
            confmat[num_indcls:num_indcls+num_oodcls, rlabel] += 1.0
        else:
            # known --> known
            confmat[plabel, rlabel] += 1.0
    if know_ood_labels:
        for rlable, plabel, uncertain in zip(ood_labels, ood_results, ood_uncertainties):
            if uncertain > threshold:
                # unknown --> unknown
                confmat[num_indcls:num_indcls+num_oodcls, num_indcls+rlable] += 1.0
            else:
                # unknown --> known
                confmat[plabel, num_indcls+rlable] += 1.0
    else:
        for plabel, uncertain in zip(ood_results, ood_uncertainties):
            if uncertain > threshold:
                # unknown --> unknown
                confmat[num_indcls:num_indcls+num_oodcls, num_indcls:num_indcls+num_oodcls] += 1 / (num_oodcls)
            else:
                # unknown --> known
                confmat[plabel, num_indcls:num_indcls+num_oodcls] += 1
    if normalize:
        confmat = confmat / confmat.sum()
        confmat = np.nan_to_num(confmat)
    return confmat


def plot_confmat(confmat, know_ood_labels=False):
    plt.figure(figsize=(4,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    plt.imshow(confmat, cmap='hot')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    save_file = args.save_file[:-4] + '_knownOOD.png' if know_ood_labels else args.save_file
    plt.savefig(save_file)
    plt.savefig(save_file[:-4] + '.pdf')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--ood_result', help='the result file of ood detection')
    parser.add_argument('--uncertain_thresh', type=float, default=0.0001, help='the threshold value for prediction')
    parser.add_argument('--save_file', help='the image file path of generated confusion matrix')
    args = parser.parse_args()

    results = np.load(args.ood_result, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    # result path
    result_path = os.path.dirname(args.save_file)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # OOD classes are unknown
    confmat1 = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                                ood_labels, ood_results, ood_uncertainties,
                                args.uncertain_thresh, know_ood_labels=False)
    plot_confmat(confmat1, know_ood_labels=False)

    # OOD classes are known
    confmat2 = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                                ood_labels, ood_results, ood_uncertainties,
                                args.uncertain_thresh, know_ood_labels=True)
    plot_confmat(confmat2, know_ood_labels=True)

    # save the confusion matrix for further analysis
    np.savez(args.save_file[:-4], confmat_unknown_ood=confmat1, confmat_known_ood=confmat2)
    votes_ind = np.sum(confmat1[:101, 101:], axis=1)
    print("Top-10 false positive IND classes: ", np.argsort(votes_ind)[-10:])

    votes_ood = np.sum(confmat1[101:, :101], axis=1)
    print("Top-10 false negative IND classes: ", np.argsort(votes_ood)[-10:])