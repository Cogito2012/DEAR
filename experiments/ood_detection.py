import argparse
import os
import os.path as osp
import torch
import mmcv
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter
from operator import itemgetter
from mmaction.datasets.pipelines import Compose
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
import numpy as np
from scipy.special import xlogy
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--uncertainty', choices=['BALD', 'Entropy', 'EDL'], help='the uncertainty estimation method')
    parser.add_argument('--forward_pass', type=int, default=10, help='the number of forward passes')
    # data config
    parser.add_argument('--ind_data', help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', help='the split file of out-of-distribution testing data')
    # env config
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--result_tag', help='result file tag')
    args = parser.parse_args()
    return args


def update_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_stochastic_inference(model, data_loader, npass=10):
    # run inference
    model = MMDataParallel(model, device_ids=[0])
    all_uncertainties, all_results, all_gts = [], [], []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        all_scores = []
        with torch.no_grad():
            for n in range(npass):
                # set new random seed
                update_seed(n * 1234)
                scores = model(return_loss=False, **data)
                # gather results
                all_scores.append(np.expand_dims(scores, axis=-1))
        all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        # compute the uncertainty
        uncertainty = compute_uncertainty(all_scores, method=args.uncertainty)
        all_uncertainties.append(uncertainty)

        # compute the predictions and save labels
        mean_scores = np.mean(all_scores, axis=-1)
        preds = np.argmax(mean_scores, axis=1)
        all_results.append(preds)

        labels = data['label'].numpy()
        all_gts.append(labels)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return all_uncertainties, all_results, all_gts


def run_evidence_inference(model, data_loader, evidence='exp'):
    # set new random seed
    update_seed(1234)
    # get the evidence function
    if evidence == 'relu':
        from mmaction.models.losses.edl_loss import relu_evidence as get_evidence
    elif evidence == 'exp':
        from mmaction.models.losses.edl_loss import exp_evidence as get_evidence
    elif evidence == 'softplus':
        from mmaction.models.losses.edl_loss import softplus_evidence as get_evidence
    else:
        raise NotImplementedError
    num_classes = model.cls_head.num_classes

    # run inference
    model = MMDataParallel(model, device_ids=[0])
    all_uncertainties, all_results, all_gts = [], [], []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            output = model(return_loss=False, **data)
            evidence = get_evidence(torch.from_numpy(output))
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
        all_uncertainties.append(uncertainty.numpy())
        # compute the predictions and save labels
        preds = np.argmax(scores.numpy(), axis=1)
        all_results.append(preds)

        labels = data['label'].numpy()
        all_gts.append(labels)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return all_uncertainties, all_results, all_gts


def run_inference(model, datalist_file, npass=10):
    # switch config for different dataset
    cfg = model.cfg
    cfg.data.test.ann_file = datalist_file
    cfg.data.test.data_prefix = os.path.join(os.path.dirname(datalist_file), 'videos')
    evidence = cfg.get('evidence', 'exp')

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False,
        pin_memory=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if not args.uncertainty == 'EDL':
        all_uncertainties, all_results, all_gts = run_stochastic_inference(model, data_loader, npass)
    else:
        all_uncertainties, all_results, all_gts = run_evidence_inference(model, data_loader, evidence)
    return all_uncertainties, all_results, all_gts


def main():

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=False)
    cfg = model.cfg
    if not args.uncertainty == 'EDL':
        # use dropout in testing stage
        if cfg.model.cls_head.type in ['I3DHead', 'TPNHead']:
            model.apply(apply_dropout)
        if cfg.model.cls_head.type in ['I3DBNNHead', 'TPNBNNHead']:
            model.test_cfg.npass = 1
    # set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    
    result_file = os.path.join('./experiments', args.result_tag + '_result.npz')
    if not os.path.exists(result_file):
        # prepare result path
        result_dir = os.path.dirname(result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # run inference (OOD)
        ood_uncertainties, ood_results, ood_labels = run_inference(model, args.ood_data, npass=args.forward_pass)
        # run inference (IND)
        ind_uncertainties, ind_results, ind_labels = run_inference(model, args.ind_data, npass=args.forward_pass)
        # save
        np.savez(result_file[:-4], ind_unctt=ind_uncertainties, ood_unctt=ood_uncertainties, 
                                   ind_pred=ind_results, ood_pred=ood_results,
                                   ind_label=ind_labels, ood_label=ood_labels)
    else:
        results = np.load(result_file, allow_pickle=True)
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
        ind_results = results['ind_pred']  # (N1,)
        ood_results = results['ood_pred']  # (N2,)
        ind_labels = results['ind_label']
        ood_labels = results['ood_label']
    # visualize
    dataName_ind = args.ind_data.split('/')[-2].upper()
    dataName_ood = args.ood_data.split('/')[-2].upper()
    plt.figure(figsize=(5,4))  # (w, h)
    plt.hist([ind_uncertainties, ood_uncertainties], 50, 
            density=True, histtype='bar', color=['blue', 'red'], 
            label=['in-distribution (%s)'%(dataName_ind), 'out-of-distribution (%s)'%(dataName_ood)])
    plt.legend(prop={'size': 10})
    plt.xlabel('%s Uncertainty'%(args.uncertainty))
    plt.ylabel('density')
    plt.tight_layout()
    plt.savefig(os.path.join('./experiments', args.result_tag + '_distribution.png'))

if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)

    main()
