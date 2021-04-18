#!/usr/bin/env python3

import os

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from skimage import io
import pickle
import pandas as pd

import utils as utls

if __name__ == "__main__":

    p = configargparse.ArgParser()
    p.add('--results-path', required=True)
    p.add('--test-ratio', default=0.2)
    p.add('--im-path', default='../dataset/images')
    p.add('--feat-path', default='../feats')
    p.add('--label-path', default='../dataset/labels')
    p.add('--feat-step', default=16, type=int)
    p.add('--nms-thr-rel', default=0.8, type=float)
    p.add('--nms-max-num-peaks', default=80, type=int)
    p.add('--do-prevs', default=False, action='store_true')

    cfg = p.parse_args()

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    samplers, test_sampler = utls.make_train_val_test(samplers,
                                                      cfg.test_ratio,
                                                      n_folds=1)
    samplers = samplers[0]
    max_n_peaks = np.max([s.get_no_aug_pos().sum() for s in samplers['train']])

    with open(os.path.join(cfg.results_path, 'clf.p'), 'rb') as f:
        clf = pickle.load(f)

    if cfg.do_prevs:
        print('saving previews...')
        out_path = os.path.join(cfg.results_path, 'prevs')
        if not os.path.exists(out_path):
            print('creating dir ', out_path)
            os.makedirs(out_path)

        pbar = tqdm.tqdm(total=len(test_sampler))
        for i, s in enumerate(test_sampler):
            peaks = utls.predict(clf, [s],
                                 cfg.feat_step,
                                 max_num_peaks=max_n_peaks,
                                 im_shape=io.imread(
                                     samplers['train'][0].im_path).shape,
                                 thr_abs=0.5)
            import pdb
            pdb.set_trace()  ## DEBUG ##
            im = io.imread(s.im_path)
            plt.imshow(im, cmap='gray')
            plt.plot(peaks[0][:, 1], peaks[0][:, 0], 'mo')
            if s._has_pos():
                plt.plot(
                    (s.get_no_aug_xy()[:, 0] * (im.shape[1] - 1)).astype(int),
                    (s.get_no_aug_xy()[:, 1] * (im.shape[0] - 1)).astype(int),
                    'go')

            plt.title('true: {}, found: {}'.format(s.pos.size,
                                                   peaks[0].shape[0]))
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, 'test_{:04d}.png'.format(i)))
            plt.close()
            pbar.update(1)
        pbar.close()

    # C = ytrue.sum()
    # Chat = ypred.sum()

    # print('C: ', C)
    # print('C hat: ', Chat)
    # print('Delta: ', np.abs(C - Chat))
    # print('Delta_norm: ', np.abs(C - Chat) / C)
