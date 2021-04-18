#!/usr/bin/env python3

import configargparse
from skimage import io
import matplotlib.pyplot as plt
import utils as utls
import os
import pickle
import numpy as np
import glob
import utils as utls
import tqdm

if __name__ == "__main__":

    p = configargparse.ArgParser()
    p.add('--results-path', required=True)
    p.add('--tau', required=True, type=float)
    p.add('--test-ratio', default=0.2)
    p.add('--im-path', default='../dataset/images')
    p.add('--feat-path', default='../feats')
    p.add('--label-path', default='../dataset/labels')
    p.add('--step', default=16, type=int)
    p.add('--nms-thr-rel', default=0.8, type=float)
    p.add('--nms-max-num-peaks', default=80, type=int)
    p.add('--do-prevs', default=False, action='store_true')

    cfg = p.parse_args()

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    _, test_sampler = utls.make_train_val_test(samplers, cfg.test_ratio)

    clfs = utls.load_clfs(cfg.results_path)

    # get classifier of last iteration
    clf = clfs[-1]['clf']

    xtest = np.concatenate([s.descs for s in test_sampler])
    ytrue = np.concatenate(
        [s.get_no_aug_pos().astype(bool) for s in test_sampler])
    ypred = clf.predict_proba(xtest)[:, 1] > cfg.tau

    if cfg.do_prevs:
        print('saving previews...')
        out_path = os.path.join(cfg.results_path, 'prevs')
        if not os.path.exists(out_path):
            print('creating dir ', out_path)
            os.makedirs(out_path)

        pbar = tqdm.tqdm(total=len(test_sampler))
        for i, s in enumerate(test_sampler):
            xy_test = s.xy
            yhat_test = clf.predict_proba(s.descs)[:, 1]
            im_test = io.imread(s.im_path)
            peaks = utls.get_peaks(xy_test,
                                   yhat_test,
                                   cfg.step,
                                   cfg.nms_max_num_peaks,
                                   im_test.shape,
                                   thr_abs=cfg.tau)
            plt.imshow(im_test, cmap='gray')
            plt.plot(peaks[:, 1], peaks[:, 0], 'mo')
            if s._has_pos():
                plt.plot((s.get_no_aug_xy()[:, 0] *
                          (im_test.shape[1] - 1)).astype(int),
                         (s.get_no_aug_xy()[:, 1] *
                          (im_test.shape[0] - 1)).astype(int), 'go')

            plt.title('true: {}, found: {}'.format(s.pos.size, peaks.shape[0]))
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, 'test_{:04d}.png'.format(i)))
            plt.close()
            pbar.update(1)
        pbar.close()

    C = ytrue.sum()
    Chat = ypred.sum()

    print('C: ', C)
    print('C hat: ', Chat)
    print('Delta: ', np.abs(C - Chat))
    print('Delta_norm: ', np.abs(C - Chat) / C)
