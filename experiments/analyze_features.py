import argparse
import os
import torch
from mmcv.parallel import collate, scatter
from mmaction.datasets.pipelines import Compose
from mmaction.apis import init_recognizer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test')
    # model config
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--known_split', help='the split file path of the knowns')
    parser.add_argument('--unknown_split', help='the split file path of the unknowns')
    parser.add_argument('--result_file', help='the result file path')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def inference_recognizer(model, video_path):
    """Inference a video with the detector.

    Args:
        model (nn.Module): The loaded recognizer.
        video_path (str): The video file path/url or the rawframes directory
            path. If ``use_frames`` is set to True, it should be rawframes
            directory path. Otherwise, it should be video file path.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data (by default, we use videodata)
    start_index = cfg.data.test.get('start_index', 0)
    data = dict(filename=video_path, label=-1, start_index=start_index, modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        feat_blob = model(return_loss=False, get_feat=True, **data) # (num_clips * num_crops, 2048, 1, 8, 8)
        # spatial average pooling 
        kernel_size = (1, feat_blob.size(-2), feat_blob.size(-1))
        avg_pool2d = torch.nn.AvgPool3d(kernel_size, stride=1, padding=0)
        feat_clips = avg_pool2d(feat_blob).view(feat_blob.size(0), feat_blob.size(1))  # (num_clips * num_crops, 2048)
        # get the mean features of all clips and crops
        feat_final = torch.mean(feat_clips, dim=0).cpu().numpy()  # (2048,)
    return feat_final

def extract_feature(video_files):
    
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=False)
    cfg = model.cfg
    torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    X = []
    for videofile in tqdm(video_files, total=len(video_files), desc='Extract Feature'):
        feature = inference_recognizer(model, videofile)  # (2048,)
        X.append(feature)
    
    return X



if __name__ == '__main__':

    args = parse_args()
    # assign the desired device.
    device = torch.device(args.device)

    known_classes = {'ParallelBars': 0, 'UnevenBars': 1, 'Diving': 2, 'Surfing': 3}
    # known_classes = {'ParallelBars': 0, 'UnevenBars': 1}
    known_data = []
    labels = []
    video_dir = os.path.join(os.path.dirname(args.known_split), '..', 'videos')
    with open(args.known_split, 'r') as f:
        for line in f.readlines():
            clsname, videoname = line.strip().split('/')
            if clsname in known_classes.keys():
                videofile = os.path.join(video_dir, line.strip())
                known_data.append(videofile)
                labels.append(known_classes[clsname])
    num_knowns = len(known_data)
    
    unknown_data = []
    video_dir = os.path.join(os.path.dirname(args.unknown_split), '..', 'videos')
    with open(args.unknown_split, 'r') as f:
        for line in f.readlines():
            videofile = os.path.join(video_dir, line.strip().split(' ')[0])
            unknown_data.append(videofile)
    # num_unknowns = int(num_knowns / len(known_classes))
    num_unknowns = num_knowns
    assert len(unknown_data) > num_unknowns
    inds = np.random.choice(len(unknown_data), num_unknowns, replace=False)
    unknown_data = [unknown_data[i] for i in inds]
    labels += [len(known_classes)] * num_unknowns
    labels = np.vstack(labels)
    open_classes = {**known_classes, 'Unknowns': len(known_classes)}

    # extracting the feature
    X = extract_feature(known_data + unknown_data)
    X = np.vstack(X)

    # run tSNE
    print('running tSNE...')
    Y = TSNE(n_components=2).fit_transform(X)
    for k, v in open_classes.items():
        inds = np.where(labels == v)[0]
        plt.scatter(Y[inds, 0], Y[inds, 1], 15, label=k)
    plt.legend()
    # plt.show()
    plt.savefig(args.result_file)
    print('Done!')