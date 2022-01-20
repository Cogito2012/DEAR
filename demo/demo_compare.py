from distutils.command.config import config
from email.mime import base
import os
from select import select
from turtle import update
from cv2 import threshold
import numpy as np
from numpy.core.fromnumeric import size
import cv2
import torch
from experiments.get_threshold import update_seed, compute_uncertainty
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import CompositeVideoClip, ImageSequenceClip, TextClip, VideoFileClip
import sys



def read_mapping(mapfile):
    assert os.path.exists(mapfile), 'Mapping file does not exist! %s'%(mapfile)
    mapping_dict = dict()
    with open(mapfile, 'r') as f:
        for line in f.readlines():
            cls_id = int(line.strip().split(' ')[0]) - 1 # class_ID starts with 0
            cls_name = line.strip().split(' ')[1]
            mapping_dict.update({cls_id: cls_name})
    return mapping_dict


def read_list(list_file, mapping):
    assert os.path.exists(list_file), 'List file does not exist! %s'%(list_file)
    videos_path = os.path.join(os.path.dirname(list_file), 'videos')
    video_list = dict()
    for k, v in mapping.items():
        video_list.update({k: []})
    with open(list_file, 'r') as f:
        for line in f.readlines():
            vid_file = os.path.join(videos_path, line.strip().split(' ')[0])
            gt_cls = int(line.strip().split(' ')[1])
            video_list[gt_cls].append(vid_file)
    return video_list


def inference_recognizer(model, video_path, npass=1):
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
    assert npass >= 1, "invalid number of forward passes!"
    with torch.no_grad():
        if npass > 1:
            scores = []
            for n in range(npass):
                update_seed(n * 1234)  # manually update the seed for randomness
                s = model(return_loss=False, **data)[0]  # (K,)
                scores.append(np.expand_dims(s, axis=-1))
            scores = np.concatenate(scores, axis=-1)  # (K, N)
        else:
            # single forward pass
            scores = model(return_loss=False, **data)[0]  # (K,)
    return scores


def evidential_prediction(logits):
    # get evidence
    evidence = np.exp(np.clip(logits, -10, 10))   # (K,)
    # get predicted class_ID
    pred_cls = int(np.argmax(evidence))  # scalar, int()
    # get uncertainty
    alpha = evidence + 1
    S = np.sum(alpha)  # scalar
    uncertainty = logits.size / S  # scalar, vacuity uncertainty
    return pred_cls, uncertainty, evidence


def read_video(video_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    while (ret):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_data.append(frame)
        ret, frame = cap.read()
    return video_data


def create_gif(outfile, fig, video_data, fps=30):
    fig.canvas.draw()
    image_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_fig = image_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    h_fig, w_fig = image_fig.shape[:2]  # (250, 2000)
    # reshape it to align with the width of video frames

    height, width = video_data[0].shape[:2]
    
    h_align = max(min(int(width * h_fig // w_fig), height), 1)
    image_fig = cv2.resize(image_fig, (width, h_align))

    last_frame = video_data[-1]
    last_frame[height - h_align:] = cv2.addWeighted(last_frame[height - h_align:], 0.3, image_fig, 0.7, 0)
    vis_data = video_data + [last_frame] * fps * 5  # last for 5 seconds after the video finished playing

    video_clips = ImageSequenceClip(vis_data, fps=fps)
    video_clips.write_gif(outfile)


def plot_evidence(gt_label, pred_label, ood_ratio, evidence, max_evidence=20000, fontsize=25):
    """ video_data: list(), size of each element is (H, W, C)
        pred_cls: int(), scalar value
        ood_ratio: float(), scalar value
        evidence: ndarray(), size = (K,)
        mapping: dict(), length=K
    """
    
    num_cls = len(evidence)
    # class_names = list(mapping.values())
    class_ids = range(num_cls)
    # produce the video frames of evidence diagram
    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=300)  # by default, dpi=100, pixel_width=4000
    ax.bar(class_ids, evidence)
    plt.xlim(0, num_cls)
    plt.ylim(0, max_evidence)
    plt.ylabel('Evidence', fontsize=fontsize)
    plt.xlabel('Known Action Types', fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.8), 'Ground Truth: %s'%(gt_label), color='y', fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.7), 'Prediction: %s'%(pred_label), color='r', fontsize=fontsize)
    plt.text(10, int(max_evidence * 0.6), 'OOD Ratio: {:.1%}'.format(ood_ratio), color='g', fontsize=fontsize)
    plt.tight_layout()
    return fig


def plot_uncertainty(gt_label, pred_label, ood_ratio, confidence, max_conf=1.0, fontsize=25):
    num_cls = len(confidence)
    class_ids = range(num_cls)
    # produce the video frames of evidence diagram
    fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=300)  # by default, dpi=100, pixel_width=4000
    ax.bar(class_ids, confidence)
    plt.xlim(0, num_cls)
    plt.ylim(0, max_conf)
    plt.ylabel('Confidence', fontsize=fontsize)
    plt.xlabel('Known Action Types', fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.text(10, max_conf * 0.8, 'Ground Truth: %s'%(gt_label), color='y', fontsize=fontsize)
    plt.text(10, max_conf * 0.7, 'Prediction: %s'%(pred_label), color='r', fontsize=fontsize)
    plt.text(10, max_conf * 0.6, 'OOD Ratio: {:.1%}'.format(ood_ratio), color='g', fontsize=fontsize)
    plt.tight_layout()
    return fig
    

def select_videos(test_data, selected=None):
    selected_files, gt_classes = [], []
    if selected is not None:
        # find the files from the selected filename list
        for gt_cls, video_list in test_data.items():
            for vid_file in video_list:
                if any(filename in vid_file for filename in selected):
                    selected_files.append(vid_file)
                    gt_classes.append(gt_cls)
    else:
        for gt_cls, video_list in test_data.items():
            # for each class, we randomly select 1 videos
            videos_keep = np.random.choice(video_list, size=1, replace=False).tolist()
            for vid_file in videos_keep:
                selected_files.append(vid_file)
                gt_classes.append(gt_cls)
    return selected_files, gt_classes


def run_visualization(result_dir, test_data, model, threshold, mapping_open, mapping_unknown=None, selected=None):
    # select video files for visualization
    selected_files, gt_classes = select_videos(test_data, selected=selected)
    # visualize
    for vid_file, gt_cls in zip(selected_files, gt_classes):
        if baseline == 'dear':
            # get the NN output logits
            scores = inference_recognizer(model, vid_file)  # (K,) ndarray
            pred_cls, uncertainty, evidence = evidential_prediction(scores)
        elif baseline == 'bnn':
            scores = inference_recognizer(model, vid_file, npass=10)  # (K, 10), 10 forward passes
            uncertainty = compute_uncertainty(np.expand_dims(scores, axis=0))[0]  # scalar
            uncertainty = np.fabs(uncertainty)
            conf_mean = np.mean(scores, axis=-1)
            pred_cls = int(np.argmax(conf_mean))
        if uncertainty > threshold:
            pred_cls = scores.shape[0]  # K, predicted as the unknown
        # visualization
        video_data = read_video(vid_file)
        real_class_name = mapping_open[gt_cls] if mapping_unknown is None else mapping_unknown[gt_cls]
        vis_file = os.path.join(result_dir, '%s_%s.gif'%(real_class_name, vid_file.split('/')[-1].split('.')[0]))
        displayed_class_name = mapping_open[gt_cls] if mapping_unknown is None else 'Unknown'
        if baseline == 'dear':
            fig = plot_evidence(displayed_class_name, mapping_open[pred_cls], uncertainty / threshold, evidence)
        elif baseline == 'bnn':
            fig = plot_uncertainty(displayed_class_name, mapping_open[pred_cls], uncertainty / threshold, conf_mean)
        # create GIF output
        create_gif(vis_file, fig, video_data)
        plt.close('all')
            

def apply_dropout(m):
    # set the dropout layer in training status (with randomness)
    if type(m) == torch.nn.Dropout:
        m.train()


def set_deterministic(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    # input configurations and model weights
    config_path = 'configs/recognition/slowfast/inference_slowfast_{}.py'  # we use slowfast as an example
    weight_path = 'work_dirs/slowfast/finetune_ucf101_slowfast_{}/latest.pth'
    configs = {
        'dear': config_path.format('enn'),
        'bnn': config_path.format('bnn'),
        'softmax': config_path.format('dnn'),
        'rpl': config_path.format('rpl')
    }
    weights = {
        'dear': weight_path.format('edlnokl_avuc_debias'),
        'bnn': weight_path.format('bnn'),
        'softmax': weight_path.format('dnn'),
        'rpl': weight_path.format('rpl')
    }
    thresholds = {'dear': 0.004552, 'bnn': 0.000010, 'softmax': 0.000065, 'rpl': 0.997780}
    set_deterministic(seed=123)

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(configs[baseline], weights[baseline], device=torch.device('cuda:0'), use_frames=False)
    if baseline == 'dear':
        # make sure the outputs of the mode are the NN logits
        assert model.cfg.evidence == 'exp', 'Use exponential evidence by setting cfg.evidence=exp !'
        assert model.cfg.test_cfg['average_clips'] == 'score', 'Please set average_clips==score in cfg.test_cfg!'
    if baseline == 'bnn':
        assert model.cfg.test_cfg['average_clips'] == 'prob', 'Please set average_clips==prob in cfg.test_cfg!'
        model.test_cfg.npass = 1  # we will use multiple forward passes in this script later.
    if baseline == 'softmax':
        model.apply(apply_dropout)
    model.cfg.data.test.test_mode = True

    # read class mapping file
    mapping_dict = read_mapping('data/ucf101/annotations/classInd.txt')
    mapping_dict.update({max(list(mapping_dict.keys()))+1: 'Unknown'})
    # read test data list (known)
    test_known = read_list('data/ucf101/ucf101_val_split_1_videos.txt', mapping_dict)
    selected_known = ['v_Basketball_g05_c03',
                      'v_GolfSwing_g06_c03',
                      'v_IceDancing_g06_c02',
                      'v_PommelHorse_g02_c03',
                      'v_SkyDiving_g05_c05']

    result_known_dir = 'demo/ucf101_compare/{}'.format(baseline)
    os.makedirs(result_known_dir, exist_ok=True)
    run_visualization(result_known_dir, test_known, model, thresholds[baseline], mapping_dict, selected=selected_known)



    # read class mapping file
    mapping_unknown_dict = read_mapping('data/hmdb51/annotations/classInd.txt')
    # read test data list (unknown)
    test_unknown = read_list('data/hmdb51/hmdb51_val_split_1_videos.txt', mapping_unknown_dict)
    selected_unknown = ['turnles!!_cartwheel_f_cm_np1_ri_med_3',
                        'Chewing_Gum_chew_h_nm_np1_fr_med_1',
                        'Basketball_Dribbling_Tips__2_dribble_f_cm_np1_le_med_3',
                        'American_Idol_Awards_Given_to_7_Winners_at_Walt_Disney_World_hug_u_cm_np2_ba_med_3',
                        'girl_smoking_smoke_h_cm_np1_ri_goo_0']

    result_unknown_dir = 'demo/hmdb51_compare/{}'.format(baseline)
    os.makedirs(result_unknown_dir, exist_ok=True)
    run_visualization(result_unknown_dir, test_unknown, model, thresholds[baseline], mapping_dict, mapping_unknown=mapping_unknown_dict, selected=selected_unknown)


if __name__ == '__main__':
    
    baseline = sys.argv[1]
    assert baseline in ['dear', 'bnn', 'softmax','rpl']
    main()


    

            