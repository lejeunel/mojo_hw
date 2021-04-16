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
    cfg = p.parse_args()

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)

    clfs = []
    print('loading classifiers...')
    for p in sorted(glob.glob(os.path.join(cfg.results_path, '*.p'))):
        with open(p, 'rb') as f:
            clfs.append(pickle.load(f))

    chunks = np.array_split(samplers, len(clfs))

    test_fold = 0
    test_im = 0
    n_candidates = 20
    sampler = chunks[test_fold][test_im]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(sampler.get_image(), cmap='gray')
    utls.plot_bboxes(sampler.xy_pos,
                     sampler.get_image().shape,
                     ax[0],
                     color='g')

    Y, _, xy = sampler.sample(N_neg=n_candidates)
    utls.plot_bboxes(xy[Y == 0], sampler.get_image().shape, ax[0], color='m')
    ax[0].title.set_text('Random neg. mining')
    ax[0].axis('off')

    ax[1].imshow(sampler.get_image(), cmap='gray')
    utls.plot_bboxes(sampler.xy_pos,
                     sampler.get_image().shape,
                     ax[1],
                     color='g')

    preds = clfs[test_fold].predict_proba(
        sampler.descs[sampler.neg_candidates])[:, 1]
    idxs = np.argsort(preds)[-n_candidates:]

    preds = clfs[test_fold].predict_proba(
        sampler.descs[sampler.neg_candidates])[:, 1]
    idxs = np.argsort(preds)[-n_candidates:]

    utls.plot_bboxes(sampler.xy[sampler.neg_candidates[idxs]],
                     sampler.get_image().shape,
                     ax[1],
                     color='m')
    ax[1].title.set_text('Hard neg. mining')
    ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.results_path, 'mining_prev.png'), dpi=200)
