import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description='Draw histogram')
    parser.add_argument("--score_func", default='uncertainty', choices=['uncertainty', 'confidence'], help='The type of scoring function for OOD detection.')
    parser.add_argument('--uncertainty', default='EDL', choices=['BALD', 'Entropy', 'EDL'], help='the uncertainty estimation method')
    parser.add_argument('--ind_data', default='UCF-101', help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', default='HMDB-51', choices=['HMDB-51', 'MiT-v2'], help='the split file of out-of-distribution testing data')
    parser.add_argument('--result_prefix', default='temp/temp.png', help='result file prefix')
    args = parser.parse_args()
    return args

def plot_by_uncertainty(result_file, auc=80, fontsize=16):
    assert os.path.exists(result_file), 'result file not exists! %s'%(result_file)
    results = np.load(result_file, allow_pickle=True)
    ind_confidences = results['ind_conf']
    ood_confidences = results['ood_conf']
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    # visualize
    ind_uncertainties = np.array(ind_uncertainties)
    ind_uncertainties = (ind_uncertainties-np.min(ind_uncertainties)) / (np.max(ind_uncertainties) - np.min(ind_uncertainties)) # normalize
    ood_uncertainties = np.array(ood_uncertainties)
    ood_uncertainties = (ood_uncertainties-np.min(ood_uncertainties)) / (np.max(ood_uncertainties) - np.min(ood_uncertainties)) # normalize

    fig = plt.figure(figsize=(5,4))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    plt.hist([ind_uncertainties, ood_uncertainties], 50, 
            density=True, histtype='bar', color=['blue', 'red'], 
            label=['in-distribution (%s)'%(args.ind_data), 'out-of-distribution (%s)'%(args.ood_data)])
    plt.legend(fontsize=fontsize-3)
    plt.text(0.6, 6, 'AUC = %.2lf'%(auc), fontsize=fontsize-3)
    plt.xlabel('%s uncertainty'%(args.uncertainty), fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, 1.01)
    plt.ylim(0, 10.01)
    plt.tight_layout()

    result_dir = os.path.dirname(args.result_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save the figure
    plt.savefig(os.path.join(args.result_prefix + '_distribution.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig(os.path.join(args.result_prefix + '_distribution.pdf'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)



if __name__ == '__main__':

    args = parse_args()

    ############################### HMDB-51 as unknown ######################################

    args.ood_data = 'HMDB-51'
    # I3D MC Dropout on HMDB
    result_file = 'i3d/results/I3D_DNN_BALD_HMDB_result.npz'
    args.uncertainty = 'BALD'
    args.result_prefix = 'i3d/results/I3D_DNN_BALD_HMDB'
    auc = 75.07
    plot_by_uncertainty(result_file, auc=auc, fontsize=18)

    # I3D BNN SVI on HMDB
    result_file = 'i3d/results/I3D_BNN_BALD_HMDB_result.npz'
    args.uncertainty = 'BALD'
    args.result_prefix = 'i3d/results/I3D_BNN_BALD_HMDB'
    auc = 74.66
    plot_by_uncertainty(result_file, auc=auc, fontsize=18)

    # I3D DRIVE (vanilla) on HMDB
    result_file = 'i3d/results/I3D_EDLNoKL_EDL_HMDB_result.npz'
    args.uncertainty = 'EDL'
    args.result_prefix = 'i3d/results/I3D_EDLNoKL_EDL_HMDB'
    auc = 76.41
    plot_by_uncertainty(result_file, auc=auc, fontsize=18)

    # I3D DRIVE (full) on HMDB
    result_file = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_HMDB_result.npz'
    args.uncertainty = 'EDL'
    args.result_prefix = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_HMDB'
    auc = 77.08
    plot_by_uncertainty(result_file, auc=auc, fontsize=18)
    
    ############################### MiT-v2 as unknown ######################################

    args.ood_data = 'MiT-v2'
    # I3D MC Dropout on MiT-v2
    result_file = 'i3d/results/I3D_DNN_BALD_MiT_result.npz'
    args.uncertainty = 'BALD'
    args.result_prefix = 'i3d/results/I3D_DNN_BALD_MiT'
    auc = 79.14
    plot_by_uncertainty(result_file, auc=auc, fontsize=20)

    # I3D BNN SVI on MiT-v2
    result_file = 'i3d/results/I3D_BNN_BALD_MiT_result.npz'
    args.uncertainty = 'BALD'
    args.result_prefix = 'i3d/results/I3D_BNN_BALD_MiT'
    auc = 79.50
    plot_by_uncertainty(result_file, auc=auc, fontsize=20)

    # I3D DRIVE (vanilla) on MiT-v2
    result_file = 'i3d/results/I3D_EDLNoKL_EDL_MiT_result.npz'
    args.uncertainty = 'EDL'
    args.result_prefix = 'i3d/results/I3D_EDLNoKL_EDL_MiT'
    auc = 81.43
    plot_by_uncertainty(result_file, auc=auc, fontsize=20)

    # I3D DRIVE (full) on MiT-v2
    result_file = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_MiT_result.npz'
    args.uncertainty = 'EDL'
    args.result_prefix = 'i3d/results/I3D_EDLNoKLAvUCCED_EDL_MiT'
    auc = 81.54
    plot_by_uncertainty(result_file, auc=auc, fontsize=20)