#!/usr/bin/env python3
import os
import pickle

import configargparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import utils as utls


def train_rf(clf, X, Y, add_negatives=0, do_hard_mining=False):
    """
    clf: classifier
    X: nd-array of features
    Y: 1d-array with values:
        - 1: foreground
        - 0: background
        - -1: unused
    add_negatives: number of samples to sample from the unused set to use as negative
    """

    if add_negatives > 0:
        idx_unused = np.where(Y == -1)[0]
        if do_hard_mining:
            # print('predicting and picking hard negatives')
            neg_probas = clf.predict_proba(X[idx_unused])[:, 1]
            hards_idx = idx_unused[np.argsort(neg_probas)[-add_negatives:]]
            Y[hards_idx] = 0

        else:
            # print('uniform random sampling of negatives')
            neg_idx = np.random.choice(idx_unused, size=add_negatives)
            Y[neg_idx] = 0

    # compute class weights
    # class_weight = {
    #     0: (Y == 1).sum() / (Y != -1).sum(),
    #     1: (Y == 0).sum() / (Y != -1).sum()
    # }
    # clf.class_weight = class_weight

    clf.fit(X[Y != -1, :], Y[Y != -1])

    return clf, Y


def train(samplers, n_trees, min_samp_split, min_samp_leaf, max_feats, n_iters,
          n_jobs, do_hard_mining):

    Y = np.concatenate([s._build_all()[0] for s in samplers])
    X = np.concatenate([s._build_all()[1] for s in samplers])

    Np = Y.sum()
    # mask all background
    Y[Y == 0] = -1

    clf = RandomForestClassifier(warm_start=True,
                                 n_jobs=n_jobs,
                                 min_samples_split=min_samp_split,
                                 max_features=max_feats,
                                 min_samples_leaf=min_samp_leaf,
                                 n_estimators=n_trees)
    for j in range(n_iters):
        # print('iter. {}/{}'.format(j + 1,
        #                            n_iters))
        if j > 0:
            clf.n_estimators += n_trees
        clf, Y = train_rf(clf,
                          X,
                          Y,
                          add_negatives=Np,
                          do_hard_mining=False if j == 0 else do_hard_mining)

    return clf


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--results-path', required=True)
    p.add('--label-path', default='../dataset/labels')
    p.add('--im-path', default='../dataset/images')
    p.add('--feat-path', default='../feats')
    p.add('--test-ratio', default=0.2, type=float)
    p.add('--n-jobs', default=4)
    p.add('--n-iters', default=4)
    p.add('--n-trees', default=10)
    p.add('--min-samp-split', default=0.1, type=float)
    p.add('--min-samp-leaf', default=0.1, type=float)
    p.add('--max-feats', default=0.1, type=float)
    p.add('--mining-iters', default=4)
    p.add('--do-hard-mining', default=False, action='store_true')
    cfg = p.parse_args()

    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)
    samplers, _ = utls.make_train_val_test(samplers, cfg.test_ratio, n_folds=1)

    train_samplers = samplers[0]['train']

    clf = train(train_samplers, cfg.n_trees, cfg.min_samp_split,
                cfg.min_samp_leaf, cfg.max_feats, cfg.n_iters, cfg.n_jobs,
                cfg.do_hard_mining)

    out_path = os.path.join(cfg.results_path, 'clf.p')
    print('saving classifier to ', out_path)
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)
