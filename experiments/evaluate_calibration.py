import argparse, os
import numpy as np
import matplotlib.pyplot as plt


def eval_calibration(predictions, confidences, labels, M=15):
    """
    M: number of bins for confidence scores
    """
    num_Bm = np.zeros((M,), dtype=np.int32)
    accs = np.zeros((M,), dtype=np.float32)
    confs = np.zeros((M,), dtype=np.float32)
    for m in range(M):
        interval = [m / M, (m+1) / M]
        Bm = np.where((confidences > interval[0]) & (confidences <= interval[1]))[0]
        if len(Bm) > 0:
            acc_bin = np.sum(predictions[Bm] == labels[Bm]) / len(Bm)
            conf_bin = np.mean(confidences[Bm])
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin
    conf_intervals = np.arange(0, 1, 1/M)
    return accs, confs, num_Bm, conf_intervals

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--ood_result', help='the result file of ood detection')
    parser.add_argument('--M', type=int, default=15, help='The number of bins')
    parser.add_argument('--save_prefix', help='the image file path of generated calibration figure')
    args = parser.parse_args()

    results = np.load(args.ood_result, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    if 'ind_conf' not in results:
        ind_confidences = 1 - ind_uncertainties
        ood_confidences = 1 - ood_uncertainties
    else:
        ind_confidences = results['ind_conf']
        ood_confidences = results['ood_conf']

    # result path
    result_path = os.path.dirname(args.save_prefix)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    accs, confs, num_Bm, conf_intervals = eval_calibration(ind_results, 1-ind_uncertainties, ind_labels, M=args.M)

    # compute Expected Calibration Error (ECE)
    ece = np.sum(np.abs(accs - confs) * num_Bm / np.sum(num_Bm))
    print('The IND ECE result: %.3lf'%(ece))

    # plot the ECE figure
    fig, ax = plt.subplots(figsize=(4,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    plt.bar(conf_intervals, accs, width=1/args.M, linewidth=1, edgecolor='k', align='edge', label='Outputs')
    plt.bar(conf_intervals, np.maximum(0, conf_intervals - accs), bottom=accs, color='y', width=1/args.M, linewidth=1, edgecolor='k', align='edge', label='Gap')
    plt.text(0.1, 0.6, 'ECE=%.4f'%(ece), fontsize=fontsize)
    add_identity(ax, color='r', ls='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('confidence', fontsize=fontsize)
    plt.ylabel('accuracy', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(args.save_prefix + '_ind.png')
    plt.savefig(args.save_prefix + '_ind.pdf')

