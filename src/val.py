#!/usr/bin/env python3
import configargparse
import os
import utils as utls
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
"""
Compute accuracy scores on trained models to select best threshold on output
"""

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--label-path', default='../dataset/labels')
    p.add('--im-path', default='../dataset/images')
    p.add('--results-path', default='../results')
    p.add('--feat-path', default='../feats')
    p.add('--folds', default=4)
    p.add('--test-ratio', default=0.1)
    p.add('--tau-start', default=0.65)
    p.add('--tau-end', default=0.8)
    p.add('--clf-iter', default=4)
    p.add('--n-tau', default=50)

    cfg = p.parse_args()

    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    folds, _ = utls.make_train_val_test(samplers,
                                        cfg.test_ratio,
                                        n_folds=cfg.folds)
    clfs = utls.load_clfs(cfg.results_path)

    tau = np.linspace(cfg.tau_start, cfg.tau_end, cfg.n_tau)
    scores = []

    print('Running {}-fold cross-validation on {} threshold values'.format(
        cfg.folds, len(tau)))

    pbar = tqdm(total=len(tau) * len(folds))
    for i, fold in enumerate(folds):
        clf = [
            c for c in clfs
            if ((c['fold'] == i) and (c['iter'] == cfg.clf_iter - 1))
        ][0]['clf']

        xval = np.concatenate([s.descs for s in fold['val']])
        yval = np.concatenate(
            [s.get_no_aug_pos().astype(bool) for s in fold['val']])
        yhat = clf.predict_proba(xval)[:, 1]
        for t in tau:

            count_y = yval.sum()
            count_ypred = (yhat > t).sum()
            s = np.abs(count_y - count_ypred)

            scores.append({
                'tau': t,
                'fold': i,
                'abs_diff': s,
                'count_y': count_y,
                'count_ypred': count_ypred
            })
            pbar.update(1)

    pbar.close()

    out_path = os.path.join(cfg.results_path, 'crossval_scores.p')
    print('saving scores to ', out_path)

    with open(out_path, 'wb') as f:
        pickle.dump(scores, f)
