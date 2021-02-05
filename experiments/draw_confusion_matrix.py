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
    confmat = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                                ood_labels, ood_results, ood_uncertainties,
                                args.uncertain_thresh, know_ood_labels=False)
    plt.figure()
    plt.imshow(confmat, cmap='hot')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.save_file)
    plt.close()

    # OOD classes are known
    confmat = confusion_maxtrix(ind_labels, ind_results, ind_uncertainties,
                                ood_labels, ood_results, ood_uncertainties,
                                args.uncertain_thresh, know_ood_labels=True)
    plt.figure()
    plt.imshow(confmat, cmap='hot')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.save_file[:-4] + '2.png')
    plt.close()

    # ind_meanUs, ood_meanUs = [], []
    # for c in range(101):
    #     # average uncertainties of IND samples on IND class
    #     ids = np.where(ind_results == c)[0]
    #     ind_meanU = np.mean(ind_uncertainties[ids]) - 0.49
    #     ind_meanUs.append(ind_meanU)
    #     # average uncertainties of OOD samples on IND class
    #     ids = np.where(ood_results == c)[0]
    #     ood_meanU = np.mean(ood_uncertainties[ids]) - 0.49
    #     ood_meanUs.append(ood_meanU)
    # # plot the bar chart
    # plt.figure(figsize=(8,4))  # (w, h)
    # width = 0.35
    # plt.bar(np.arange(101) - width/2, ind_meanUs, width, label='UCF-101')
    # plt.bar(np.arange(101) + width/2, ood_meanUs, width, label='HMDB-51')
    # plt.xlabel('UCF-101 Classes')
    # plt.ylabel('Uncertainties')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(args.save_file)
