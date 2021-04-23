#!/usr/bin/env python3
import os
import pickle

import configargparse
import numpy as np
from skimage import io
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

import utils as utls
from train import train
"""
Perform cross-validation on "tree complexity" and threshold value of Random Forest classifier
"""

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--label-path', default='../dataset/labels')
    p.add('--im-path', default='../dataset/images')
    p.add('--results-path', default='../results')
    p.add('--feat-path', default='../feats')
    p.add('--feat-step', default=16, type=int)

    p.add('--folds', default=4, type=int)
    p.add('--n-iters', default=4, type=int)
    p.add('--n-jobs', default=4, type=int)
    p.add('--do-hard-mining', default=False, action='store_true')
    p.add('--test-ratio', default=0.2, type=float)
    p.add('--clf-iter', default=4)
    p.add('--n-trees', default=10, type=int)

    # cross-val params
    p.add('--max-feats-start', default=0.05, type=float)
    p.add('--max-feats-end', default=0.2, type=float)
    p.add('--n-max-feats', default=4, type=int)

    p.add('--min-samp-split-start', default=0.01, type=float)
    p.add('--min-samp-split-end', default=0.05, type=float)
    p.add('--n-min-samp-split', default=1, type=int)

    p.add('--tau-start', default=0.5, type=float)
    p.add('--tau-end', default=0.8, type=float)
    p.add('--n-tau', default=4, type=int)

    cfg = p.parse_args()

    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    folds, _ = utls.make_train_val_test(samplers,
                                        cfg.test_ratio,
                                        n_folds=cfg.folds)
    print('Running {}-fold cross-validation'.format(cfg.folds))

    param_grid = ParameterGrid({
        'min_samp_split':
        np.linspace(cfg.min_samp_split_start, cfg.min_samp_split_end,
                    cfg.n_min_samp_split),
        'max_feats':
        np.linspace(cfg.max_feats_start, cfg.max_feats_end, cfg.n_max_feats),
        'tau':
        np.linspace(cfg.tau_start, cfg.tau_end, cfg.n_tau),
    })

    scores = []

    pbar = tqdm(total=len(folds) * len(param_grid))
    for params in param_grid:
        for i, fold in enumerate(folds):
            clf = train(fold['train'], cfg.n_trees, params['min_samp_split'],
                        1, params['max_feats'], cfg.n_iters, cfg.n_jobs,
                        cfg.do_hard_mining)

            c = np.array([s.get_no_aug_pos().sum() for s in fold['val']])

            peaks = utls.predict(
                clf,
                fold['val'],
                cfg.feat_step,
                max_num_peaks=np.max(
                    [s.get_no_aug_pos().sum() for s in fold['train']]),
                im_shape=io.imread(fold['val'][0].im_path).shape,
                thr_abs=params['tau'])
            c_hat = np.array([p.shape[0] for p in peaks])

            mae = np.abs(c - c_hat).mean()
            mare = (np.abs(c - c_hat) / c).mean()
            scores += [{
                'fold': i,
                'n_trees': cfg.n_trees,
                'min_samp_split': params['min_samp_split'],
                'tau': params['tau'],
                'max_feats': params['max_feats'],
                'mae': mae,
                'mare': mare,
                'count_y': c.sum(),
                'count_ypred': c_hat.sum()
            }]

            # pbar.set_description('f {}, '.format(loss_))
            pbar.update(1)

    pbar.close()

    out_path = os.path.join(cfg.results_path, 'crossval_scores.p')
    print('saving scores to ', out_path)

    with open(out_path, 'wb') as f:
        pickle.dump(scores, f)
