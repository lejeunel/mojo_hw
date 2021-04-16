#!/usr/bin/env python3
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from sampler import Sampler
import glob
from skimage import io
import os
import re
from tqdm import tqdm
import pickle


def make_train_val_test(samplers, test_ratio, n_folds=4):
    test_trainval_samplers = np.array_split(samplers,
                                            [int(len(samplers) * test_ratio)])
    train_val_samplers = test_trainval_samplers[1]
    test_samplers = test_trainval_samplers[0]

    chunks = np.array_split(train_val_samplers, n_folds)

    folds = []
    for n in range(n_folds):
        train_fold = [c for k, c in enumerate(chunks) if k != n]
        train_fold = [item for sublist in train_fold for item in sublist]
        dict_ = {'train': train_fold, 'val': chunks[n]}
        folds.append(dict_)

    return folds, test_samplers


def load_clfs(path):

    clfs = []
    print('loading classifiers...')
    for p in sorted(glob.glob(os.path.join(path, 'clf_*.p'))):
        z = re.match("clf_fold_(\d)_iter_(\d)", os.path.split(p)[-1])
        with open(p, 'rb') as f:
            clfs.append({
                'fold': int(z.group(1)),
                'iter': int(z.group(2)),
                'clf': pickle.load(f)
            })

    return clfs


def build_samplers(im_path, feat_path, label_path):
    ims_fname = sorted(glob.glob(os.path.join(im_path, '*.png')))
    feats_fname = sorted(glob.glob(os.path.join(feat_path, '*.npz')))
    labels_fname = sorted(glob.glob(os.path.join(label_path, '*.txt')))

    samplers = []

    print('Loading data and building samplers')
    pbar = tqdm(total=len(feats_fname))
    for i in range(len(feats_fname)):

        labels = read_labels(labels_fname[i])
        xy, descs = read_feats(feats_fname[i])

        sampler = Sampler(labels, xy, descs, im_path=ims_fname[i])

        samplers.append(sampler)

        pbar.update(1)

    pbar.close()

    return samplers


def read_feats(fname):
    npf = np.load(fname)
    return np.vstack((npf['x'], npf['y'])).T, npf['descs']


def read_labels(fname):

    out = np.loadtxt(fname)
    if out.size == 0:
        return np.array([[], []])

    if out.ndim == 1:
        out = out[None, ...]

    return out[:, 1:3]


def plot_bboxes(xy, shape, ax, w=0.04, h=0.06, color='b'):

    w_ = int(w * np.max(shape))
    h_ = int(h * np.max(shape))

    for x_, y_ in xy:
        # Create a Rectangle patch
        x_ = x_ * shape[1]
        y_ = y_ * shape[0]
        rect = patches.Rectangle((x_ - w_ // 2, y_ - h_ // 2),
                                 w_,
                                 h_,
                                 linewidth=1,
                                 edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)
