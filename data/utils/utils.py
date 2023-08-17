# -*- coding: utf-8 -*-
"""
| **@created on:** 7/23/20,
| **@author:** prathyushsp, XHZGenius
| **@version:** v0.0.2
|
| **Description:** Moved some functions to other places. 
|    E.g. get_model is moved to trained_models/__init__.py. 
| 
|
| **Sphinx Documentation Status:** 
"""

import numpy as np
from torch.autograd import Variable
import torch
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
import math
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

SNS_CMAP = ListedColormap(sns.light_palette('red').as_hex())

def get_md5_checksum(file_list):
    md5sum = []
    for file in file_list:
        hasher = hashlib.md5()
        with open(file, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        md5sum.append(hasher.hexdigest())
    return md5sum


def dual_min_max_norm(data, fixed_min=None, fixed_max=None):
    pos_indices = np.argwhere(data > 0)
    pos_features = data[pos_indices]
    neg_indices = np.argwhere(data < 0)
    neg_features = data[neg_indices]

    pos_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    pos_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    pos_features = (pos_features - pos_features_min) / (pos_features_max - pos_features_min)

    neg_features = np.abs(neg_features)
    neg_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    neg_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    neg_features = (neg_features - neg_features_min) / (neg_features_max - neg_features_min)

    data[pos_indices] = pos_features
    data[neg_indices] = -neg_features
    return data


def print_var_stats(var):
    print(f"Min: {var.min()} ({np.argmin(var)}) | Max: {var.max()} ({np.argmax(var)}) | Var: {var.var()}")

def distance_metrics(sample_a, sample_b, def_key=''):
    dist_metrics = {}
    dist_metrics['euc'] = euclidean(sample_a, sample_b)
    dist_metrics['dtw'] = fastdtw(sample_a, sample_b)[0]
    dist_metrics['cs'] = cosine_similarity([sample_a], [sample_b])[0][0]
    return {def_key + k: v for k, v in dist_metrics.items()}

def model_metrics(model, sample, label, def_key=''):
    model_metrics_data = {}
    raw_preds = model(torch.tensor(sample, dtype=torch.float32))
    prob = torch.nn.Softmax(dim=-1)(raw_preds).numpy()
    raw_preds = raw_preds.numpy()
    model_metrics_data['label'] = label
    model_metrics_data['raw_pred_class_0'] = float(raw_preds[0])
    model_metrics_data['raw_pred_class_1'] = float(raw_preds[1])
    model_metrics_data['prob_class_0'] = float(prob[0])
    model_metrics_data['prob_class_1'] = float(prob[1])
    model_metrics_data['conf'] = np.max(prob)
    model_metrics_data['prediction'] = np.argmax(prob)
    model_metrics_data['pred_acc'] = 1 if model_metrics_data['prediction'] == label else 0
    return {def_key + k: v for k, v in model_metrics_data.items()}


def set_global_seed(seed_value):
    print(f"Setting seed ({seed_value})  . . .")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)
    random.seed(seed_value)


def get_run_configuration(args, dataset, TASK_ID):
    if args.dataset_type == 'train':
        data = dataset.train_data
        label = dataset.train_label
    elif args.dataset_type == 'test':
        data = dataset.test_data
        label = dataset.test_label
    elif args.dataset_type == 'valid':
        data = dataset.valid_data
        label = dataset.valid_label
    else:
        raise Exception(f"Unknown dataset_type : {args.dataset_type}. Supported - [train, test, representative]")
    print(f"Running on {args.dataset_type} data")

    if args.run_mode == 'single':
        ds = enumerate(zip([data[args.single_sample_id]], [label[args.single_sample_id]]))
        print(f"Running a single sample: idx {args.single_sample_id} . . .")
    elif args.run_mode == 'local':
        ds = enumerate(zip(data, label))
        print(f"Running in local mode on complete data . . .")
    else:
        print(f"Running in turing mode using slurm tasks . . .")
        if args.jobs_per_task > 0:
            args.samples_per_task = math.ceil(len(data) / args.jobs_per_task)
        ds = enumerate(
            zip(data[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + args.samples_per_task],
                label[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + args.samples_per_task]))
    return ds



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image(img):
    if len(img.shape) == 4:
        img = np.transpose(img[0], (1, 2, 0))
        return np.uint8(255 * img)
    else:
        return np.uint8(255 * img)


# def tv_norm(input, tv_beta):
#     img = input[0, 0, :]
#     row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
#     col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
#     return row_grad + col_grad

def tv_norm(signal, tv_beta):
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


def preprocess_image(img, use_cuda):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save_timeseries(mask, time_series, save_dir=None, raw_mask=None, category=None):
    mask = mask
    # mask = (mask - np.min(mask)) / (np.max(mask) + 1e-8)
    # uplt = plot_saliency_cmap(data=time_series, weights=mask, plt=plt, display=True, dataset_name=dataset, labels=algo)
    
    if raw_mask is None:
        uplt = plot_cmap(time_series, mask)
    else:
        uplt = plot_cmap_multi(time_series, norm_saliency=mask, raw_saliency=raw_mask, category=category)
    uplt.xlabel("Timesteps")
    uplt.ylabel("Value")
    uplt.show() # By XHZ
    # plt.savefig(os.path.join(save_dir, "saliency_map")) # By XHZ


def save(mask, img, blurred, save_dir, enable_wandb=False):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite(f"{save_dir}/res-perturbated.png", np.uint8(255 * perturbated))

    cv2.imwrite(f"{save_dir}/res-heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(f"{save_dir}/res-mask.png", np.uint8(255 * mask))
    cv2.imwrite(f"{save_dir}/res-cam.png", np.uint8(255 * cam))


def numpy_to_torch(img, use_cuda, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v



def plot_cmap_multi(data, norm_saliency, raw_saliency, category):
    # CMAP = ListedColormap([*sns.light_palette('red').as_hex()[::-1], "#FFFFFF" , *sns.light_palette('green').as_hex()])
    CMAP = get_cmap("PiYG")
    try:
        data = data.flatten().tolist()
        raw_saliency = -raw_saliency if category==1 else raw_saliency
        # raw_saliency = dual_min_max_norm(raw_saliency, fixed_max=1.0, fixed_min=0.0)
        timesteps = len(data)
        plt.clf()
        fig = plt.gcf()

        raw_saliency = raw_saliency.flatten()
        # print("raw saliency:", raw_saliency)
        # raw_saliency[np.argmin(raw_saliency)]=-1
        # raw_saliency[np.argmax(raw_saliency)]=1
        im = plt.imshow(raw_saliency.reshape([1, -1]), cmap=CMAP, aspect="auto", alpha=0.85,
                        extent=[0, len(raw_saliency) - 1, float(np.min([np.min(data)])) - 1e-1,
                                float(np.max([np.max(data)])) + 1e-1]
                        )
        plt.plot(data)
        plt.grid(False)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
        plt.clim(-1, 1) # Limit the range of color by XHZ
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=4)
    except Exception as e:
        print(e)
        print("Failed to generate the CMAP!")
        pass
    return plt

def plot_cmap(data, saliency):
    try:
        data = data#.cpu().detach().numpy().flatten().tolist()
        timesteps = len(data)
        plt.clf()
        fig = plt.gcf()
        im = plt.imshow(saliency.reshape([1, -1]), cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                        extent=[0, len(saliency) - 1, float(np.min([np.min(data), np.min(saliency)])) - 1e-1,
                                float(np.max([np.max(data), np.max(saliency)])) + 1e-1]
                        )
        plt.plot(data, lw=4)
        plt.grid(False)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=4)
    except Exception:
        print("Failed to generate the CMAP!")
        pass
    return plt

def plot_saliency(data, saliency): # Custom
    CMAP = get_cmap("PiYG")
    # print(data, saliency)
    data = data.flatten()
    saliency = saliency.flatten()
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.cpu().detach().numpy()
    plt.clf()
    fig = plt.gcf()
    im = plt.imshow(saliency.reshape([1, -1]), cmap=CMAP, aspect="auto", alpha=0.85,
                    extent=[-0.5, len(saliency) - 1+0.5, float(np.min([np.min(data)])) - 1e-1,
                            float(np.max([np.max(data)])) + 1e-1]
                    )
    plt.plot(data)
    plt.grid(False)
    plt.xlabel("Timestep or Frequency")
    plt.ylabel("Values")
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    plt.clim(-1, 1) # Limit the range of color
    fig.colorbar(im, cax=cax, orientation="horizontal")
    plt.tight_layout(pad=4)
    plt.show()
    pass
