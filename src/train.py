#!/usr/bin/env python3
import configargparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import utils as utls
from sampler import Sampler
import glob
import os
from sklearn.ensemble import RandomForestClassifier
import pickle


def predict_negs(clf, samplers):
    """
    Get output probabilities of all negatives
    so as to produce hard-negative mining

    clf: Classifier
    samplers: Sampler objects
    """

    n_samples = [s.neg_candidates.size for s in samplers]
    f = np.concatenate([s.descs[s.neg_candidates] for s in samplers])

    preds = clf.predict_proba(f)[:, 1]

    preds = np.split(preds, np.cumsum(n_samples))[:-1]

    # normalize to 1-sum
    preds = [p / p.sum() for p in preds]

    return preds


def train_rf(clf, samplers, sample_mode='random'):

    Y = []
    f = []

    sampling_probas = [None] * len(samplers)
    if sample_mode != 'random':
        sampling_probas = predict_negs(clf, samplers)

    for p, s in zip(sampling_probas, samplers):
        Y_, f_, _ = s.sample(p=p)
        Y.append(Y_)
        f.append(f_)

    clf.fit(np.concatenate(f), np.concatenate(Y))

    return clf


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--label-path', default='../dataset/labels')
    p.add('--im-path', default='../dataset/images')
    p.add('--results-path', default='../results')
    p.add('--feat-path', default='../feats')
    p.add('--folds', default=4)
    p.add('--n-jobs', default=4)
    p.add('--n-trees', default=100)
    p.add('--min-samp-split', default=0.05)
    p.add('--mining-iters', default=4)
    cfg = p.parse_args()

    if not os.path.exists(cfg.results_path):
        ok.makedirs(cfg.results_path)

    samplers = utls.build_samplers(cfg.im_path, cfg.feat_path, cfg.label_path)

    chunks = np.array_split(samplers, cfg.folds)

    for i in range(cfg.folds):
        # get samplers for train and validation
        train_samplers = [c for k, c in enumerate(chunks) if k != i]
        train_samplers = [
            item for sublist in train_samplers for item in sublist
        ]
        val_samplers = chunks[i]
        for j in range(cfg.mining_iters):
            print('fold {}/{}, iter. {}/{}'.format(i + 1, cfg.folds, j + 1,
                                                   cfg.mining_iters))
            if j == 0:
                # first iteration: build classifier
                clf = RandomForestClassifier(
                    verbose=False,
                    n_jobs=cfg.n_jobs,
                    min_samples_split=cfg.min_samp_split,
                    n_estimators=cfg.n_trees)
                clf = train_rf(clf, train_samplers)
            else:
                clf = train_rf(clf, train_samplers, sample_mode='weighted')

            out_path = os.path.join(cfg.results_path,
                                    'clf_fold_{}_iter_{}.p'.format(i, j))
            print('saving classifier to ', out_path)
            with open(out_path, 'wb') as f:
                pickle.dump(clf, f)

    # import matplotlib.pyplot as plt
    # _, ax = plt.subplots()
    # ax.imshow(im, cmap='gray')
    # utls.plot_bboxes(xy[Y == 1, :], im.shape, ax, color='g')
    # utls.plot_bboxes(xy[Y == 0, :], im.shape, ax, color='r')

    # plt.show()
