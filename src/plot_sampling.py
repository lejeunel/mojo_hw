#!/usr/bin/env python3

import configargparse
import matplotlib.pyplot as plt
import utils as utls
import os
import pickle
import numpy as np
import glob

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--label-path', default='../dataset/labels')
    p.add('--im-path', default='../dataset/images')
    p.add('--results-path', default='../results')
    p.add('--im-ext', default='png')
    p.add('--feat-path', default='../feats')
    p.add('--test-ratio', default=0.2, type=float)
    cfg = p.parse_args()

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    samplers, _ = utls.make_train_val_test(samplers, cfg.test_ratio, n_folds=1)

    with open(os.path.join(cfg.results_path, 'clf.p'), 'rb') as f:
        clf = pickle.load(f)

    test_im = 0
    sample = samplers[0]['train'][test_im]

    # sample hard and random negatives
    X = sample.descs
    to_add = sample.xy_pos.shape[0]
    probas = clf.predict_proba(X)[:, 1][sample.neg_candidates]
    xy_candidates = sample.xy[sample.neg_candidates, :]
    xy_hards = xy_candidates[np.argsort(probas)[-to_add:], :]
    idx_rnd = np.random.choice(sample.neg_candidates, size=to_add)
    xy_rnd = sample.xy[idx_rnd, :]
    xy_pos = sample.xy_pos
    im = sample.get_image()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].title.set_text('Random neg. mining')
    ax[0].axis('off')
    ax[0].imshow(im, cmap='gray')
    ax[0].plot((xy_pos[:, 0] * (im.shape[1] - 1)).astype(int),
               (xy_pos[:, 1] * (im.shape[0] - 1)).astype(int), 'go')
    ax[0].plot((xy_rnd[:, 0] * (im.shape[1] - 1)).astype(int),
               (xy_rnd[:, 1] * (im.shape[0] - 1)).astype(int), 'mo')

    ax[1].imshow(im, cmap='gray')
    ax[1].plot((xy_pos[:, 0] * (im.shape[1] - 1)).astype(int),
               (xy_pos[:, 1] * (im.shape[0] - 1)).astype(int), 'go')
    ax[1].plot((xy_hards[:, 0] * (im.shape[1] - 1)).astype(int),
               (xy_hards[:, 1] * (im.shape[0] - 1)).astype(int), 'mo')

    ax[1].title.set_text('Hard neg. mining')
    ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.results_path, 'mining_prev.png'), dpi=200)
