import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def softmax_curvepoints(result_file, thresh, ood_ncls, num_rand):
    assert os.path.exists(result_file), "File not found! Run baseline_i3d_softmax.py first!"
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    ind_softmax = results['ind_softmax']  # (N1, C)
    ood_softmax = results['ood_softmax']  # (N2, C)
    ind_labels = results['ind_label']  # (N1,)
    ood_labels = results['ood_label']  # (N2,)

    ind_ncls = ind_softmax.shape[1]
    ind_results = np.argmax(ind_softmax, axis=1)
    ood_results = np.argmax(ood_softmax, axis=1)
    ind_conf = np.max(ind_softmax, axis=1)
    ood_conf = np.max(ood_softmax, axis=1)

    ind_results[ind_conf < thresh] = ind_ncls  # incorrect rejection
    # open set F1 score (multi-class)
    macro_F1 = f1_score(ind_labels, ind_results, average='macro')
    macro_F1_list = [macro_F1 * 100]
    openness_list = [0]
    for n in range(ood_ncls):
        ncls_novel = n + 1
        openness = (1 - np.sqrt((2 * ind_ncls) / (2 * ind_ncls + ncls_novel))) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(ood_ncls, ncls_novel, replace=False) 
            ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
            ood_sub_labels = np.ones_like(ood_sub_results) * ind_ncls
            ood_sub_confs = np.concatenate([ood_conf[ood_labels == clsid] for clsid in cls_select])
            ood_sub_results[ood_sub_confs < thresh] = ind_ncls  # correct rejection
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average='macro')
        macro_F1 = np.mean(macro_F1_multi) * 100
        macro_F1_list.append(macro_F1)
    return openness_list, macro_F1_list

def openmax_curvepoints(result_file, ood_ncls, num_rand):
    assert os.path.exists(result_file), "File not found! Run baseline_i3d_openmax.py first!"
    results = np.load(result_file, allow_pickle=True)
    ind_openmax = results['ind_openmax']  # (N1, C+1)
    ood_openmax = results['ood_openmax']  # (N2, C+1)
    ind_labels = results['ind_label']  # (N1,)
    ood_labels = results['ood_label']  # (N2,)
    ind_results = np.argmax(ind_openmax, axis=1)
    ood_results = np.argmax(ood_openmax, axis=1)
    ind_ncls = ind_openmax.shape[1] - 1  # (C+1)-1

    # open set F1 score (multi-class)
    macro_F1 = f1_score(ind_labels, ind_results, average='macro')
    macro_F1_list = [macro_F1 * 100]
    openness_list = [0]
    for n in range(ood_ncls):
        ncls_novel = n + 1
        openness = (1 - np.sqrt((2 * ind_ncls) / (2 * ind_ncls + ncls_novel))) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(ood_ncls, ncls_novel, replace=False) 
            ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
            ood_sub_labels = np.ones_like(ood_sub_results) * ind_ncls
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average='macro')
        macro_F1 = np.mean(macro_F1_multi) * 100
        macro_F1_list.append(macro_F1)
    return openness_list, macro_F1_list


def uncertainty_curvepoints(result_file, thresh, ind_ncls, ood_ncls, num_rand):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!"
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    # open set F1 score (multi-class)
    ind_results[ind_uncertainties > thresh] = ind_ncls  # falsely rejection
    macro_F1 = f1_score(ind_labels, ind_results, average='macro')
    macro_F1_list = [macro_F1 * 100]
    openness_list = [0]
    for n in range(ood_ncls):
        ncls_novel = n + 1
        openness = (1 - np.sqrt((2 * ind_ncls) / (2 * ind_ncls + ncls_novel))) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(ood_ncls, ncls_novel, replace=False) 
            ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
            ood_sub_uncertainties = np.concatenate([ood_uncertainties[ood_labels == clsid] for clsid in cls_select])
            ood_sub_results[ood_sub_uncertainties > thresh] = ind_ncls  # correctly rejection
            ood_sub_labels = np.ones_like(ood_sub_results) * ind_ncls
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average='macro')
        macro_F1 = np.mean(macro_F1_multi) * 100
        macro_F1_list.append(macro_F1)
    return openness_list, macro_F1_list


def plot_all_curves(openness, values, line_styles, fontsize=18):
    fig = plt.figure(figsize=(8,6))  # (w, h)
    plt.rcParams["font.family"] = "Arial"
    for k, v in values.items():
        plt.plot(openness, v, line_styles[k], linewidth=2, label=k)
    plt.xlim(0, max(openness))
    plt.ylim(60, 80)
    plt.xlabel('Openness (%)', fontsize=fontsize)
    plt.ylabel('Open maF1 (%)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(np.arange(60, 81, 5), fontsize=fontsize)
    plt.grid('on')
    plt.legend(fontsize=fontsize, loc='lower left')
    plt.tight_layout()
    result_path = os.path.dirname(args.result_prefix)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plt.savefig(args.result_prefix + '_%s.png'%(args.ood_data), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(args.result_prefix + '_%s.pdf'%(args.ood_data), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)

def main_i3d():
    # SoftMax
    print('Compute Open maF1 for SoftMax...')
    result_file = 'i3d/results_baselines/openmax/I3D_OpenMax_%s_result.npz'%(args.ood_data)
    openness_softmax, maF1_softmax = softmax_curvepoints(result_file, 0.996825, args.ood_ncls, args.num_rand)

    # OpenMax
    print('Compute Open maF1 for OpenMax...')
    result_file = 'i3d/results_baselines/openmax/I3D_OpenMax_%s_result.npz'%(args.ood_data)
    openness_openmax, maF1_openmax = openmax_curvepoints(result_file, args.ood_ncls, args.num_rand)

    # RPL
    print('Compute Open maF1 for RPL...')
    result_file = 'i3d/results_baselines/rpl/I3D_RPL_%s_result.npz'%(args.ood_data)
    openness_rpl, maF1_rpl = softmax_curvepoints(result_file, 0.995178, args.ood_ncls, args.num_rand)

    # MCDropout BALD
    print('Compute Open maF1 for MC Dropout BALD...')
    result_file = 'i3d/results/I3D_DNN_BALD_%s_result.npz'%(args.ood_data)
    openness_dnn, maF1_dnn = uncertainty_curvepoints(result_file, 0.000433, args.ind_ncls, args.ood_ncls, args.num_rand)

    # BNN SVI BALD
    print('Compute Open maF1 for BNN SVI BALD...')
    result_file = 'i3d/results/I3D_BNN_BALD_%s_result.npz'%(args.ood_data)
    openness_bnn, maF1_bnn = uncertainty_curvepoints(result_file, 0.000004, args.ind_ncls, args.ood_ncls, args.num_rand)

    # # DEAR (vanilla)
    # print('Compute Open maF1 for DEAR (vanilla)...')
    # result_file = 'i3d/results/I3D_EDLNoKL_EDL_%s_result.npz'%(args.ood_data)
    # openness_enn, maF1_enn = uncertainty_curvepoints(result_file, 0.004547, args.ind_ncls, args.ood_ncls, args.num_rand)

    # DEAR (ours)
    print('Compute Open maF1 for DEAR (ours)...')
    result_file = 'i3d/results/I3D_EDLNoKLAvUCDebias_EDL_%s_result.npz'%(args.ood_data)
    openness_dear, maF1_dear = uncertainty_curvepoints(result_file, 0.004550, args.ind_ncls, args.ood_ncls, args.num_rand)

    # draw F1 curve
    line_styles = {'DEAR (ours)': 'r-', 'SoftMax': 'b-', 'RPL': 'm-', 'BNN SVI': 'c-', 'MC Dropout': 'y-', 'OpenMax': 'k-'}
    values = {'DEAR (ours)': maF1_dear, 'SoftMax': maF1_softmax, 'RPL': maF1_rpl, 'BNN SVI': maF1_bnn, 'MC Dropout': maF1_dnn, 'OpenMax': maF1_openmax}
    plot_all_curves(openness_dear, values, line_styles, fontsize=22)


def main_slowfast():
    # SoftMax
    print('Compute Open maF1 for SoftMax...')
    result_file = 'slowfast/results_baselines/openmax/SlowFast_OpenMax_%s_result.npz'%(args.ood_data)
    openness_softmax, maF1_softmax = softmax_curvepoints(result_file, 0.996825, args.ood_ncls, args.num_rand)

    # OpenMax
    print('Compute Open maF1 for OpenMax...')
    result_file = 'slowfast/results_baselines/openmax/SlowFast_OpenMax_%s_result.npz'%(args.ood_data)
    openness_openmax, maF1_openmax = openmax_curvepoints(result_file, args.ood_ncls, args.num_rand)

    # RPL
    print('Compute Open maF1 for RPL...')
    result_file = 'slowfast/results_baselines/rpl/SlowFast_RPL_%s_result.npz'%(args.ood_data)
    openness_rpl, maF1_rpl = softmax_curvepoints(result_file, 0.995178, args.ood_ncls, args.num_rand)

    # MCDropout BALD
    print('Compute Open maF1 for MC Dropout BALD...')
    result_file = 'slowfast/results/SlowFast_DNN_BALD_%s_result.npz'%(args.ood_data)
    openness_dnn, maF1_dnn = uncertainty_curvepoints(result_file, 0.000433, args.ind_ncls, args.ood_ncls, args.num_rand)

    # BNN SVI BALD
    print('Compute Open maF1 for BNN SVI BALD...')
    result_file = 'slowfast/results/SlowFast_BNN_BALD_%s_result.npz'%(args.ood_data)
    openness_bnn, maF1_bnn = uncertainty_curvepoints(result_file, 0.000004, args.ind_ncls, args.ood_ncls, args.num_rand)

    # # DEAR (vanilla)
    # print('Compute Open maF1 for DEAR (vanilla)...')
    # result_file = 'slowfast/results/SlowFast_EDLNoKL_EDL_%s_result.npz'%(args.ood_data)
    # openness_enn, maF1_enn = uncertainty_curvepoints(result_file, 0.004547, args.ind_ncls, args.ood_ncls, args.num_rand)

    # DEAR (ours)
    print('Compute Open maF1 for DEAR (ours)...')
    result_file = 'slowfast/results/SlowFast_EDLNoKLAvUCDebias_EDL_%s_result.npz'%(args.ood_data)
    openness_dear, maF1_dear = uncertainty_curvepoints(result_file, 0.004550, args.ind_ncls, args.ood_ncls, args.num_rand)

    # draw F1 curve
    line_styles = {'DEAR (ours)': 'r-', 'SoftMax': 'b-', 'RPL': 'm-', 'BNN SVI': 'c-', 'MC Dropout': 'y-', 'OpenMax': 'k-'}
    values = {'DEAR (ours)': maF1_dear, 'SoftMax': maF1_softmax, 'RPL': maF1_rpl, 'BNN SVI': maF1_bnn, 'MC Dropout': maF1_dnn, 'OpenMax': maF1_openmax}
    plot_all_curves(openness_dear, values, line_styles, fontsize=20)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare the performance of Open macroF1 against openness')
    # model config
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, default=51, choices=[51, 305], help='the number of classes in unknwon dataset')
    parser.add_argument('--ood_data', default='HMDB', choices=['HMDB', 'MiT'], help='the name of OOD dataset.')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    parser.add_argument('--result_prefix', default='../temp/F1_openness')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    np.random.seed(123)
    args = parse_args()

    # draw results on I3D
    main_i3d()

    # # draw results on TPN
    # main_slowfast()
