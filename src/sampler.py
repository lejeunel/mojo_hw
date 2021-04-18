#!/usr/bin/env python3
import numpy as np
from skimage import io
from sklearn.metrics import pairwise_distances

import utils as utls


class Sampler:
    """
    Class that samples negatives

    """
    def __init__(self,
                 xy_pos,
                 xy,
                 descs,
                 neg_dist=0.1,
                 pos_dist=0.01,
                 default_N_negs=10,
                 im_path=None):
        """
        xy_pos: ndarray with coordinates of positive patches
        xy: ndarray with coordinates of features
        descs: ndarray with each row a feature vector
        neg_min_dist: minimum relative distance that a negative can have w.r.t a positive
        pos_dist: maximum relative distance that an augmented positive can have w.r.t a positive
        default_N_negs: when image has no positives, take this amount of negatives
        """

        self.xy_pos = xy_pos
        self.xy = xy
        self.descs = descs
        self.neg_dist = neg_dist
        self.pos_dist = pos_dist
        self.default_N_negs = default_N_negs
        self.im_path = im_path

        # make sure our negatives do not overlap too much with positives
        if self._has_pos():
            pw_dist = pairwise_distances(self.xy, self.xy_pos)

            # pre-compute valid negative candidates using pairwise distance
            is_valid = (pw_dist > self.neg_dist).sum(
                axis=1) == self.xy_pos.shape[0]
            self.neg_candidates = np.where(is_valid)[0]

            # take feature matrix of positives w.r.t closest distance
            is_valid = (pw_dist <= self.pos_dist).sum(axis=1) > 0
            self.pos_candidates = np.where(is_valid)[0]
            self.pos = pw_dist.argmin(axis=0)
            self.xy_pos = self.xy[self.pos_candidates, :]
        else:
            # all are valid negative candidates
            self.neg_candidates = np.arange(self.xy.shape[0])
            self.pos_candidates = np.array([])
            self.pos = np.array([])

    def get_image(self):
        return io.imread(self.im_path)

    def get_pos(self):
        """
        Return a {0;1} vector of positives (with augmented)
        """
        y = np.zeros(self.descs.shape[0]).astype(int)
        if self.pos.size > 0:
            y[self.pos_candidates] = 1
        return y

    def get_no_aug_pos(self):
        """
        Return a {0;1} vector of annotated positives (without augmented)
        """
        y = np.zeros(self.descs.shape[0]).astype(int)
        if self.pos.size > 0:
            y[self.pos] = 1
        return y

    def get_no_aug_xy(self):
        """
        Return a xy  array of annotated positives (without augmented)
        """
        y = np.zeros(self.descs.shape[0]).astype(int)
        if self.pos.size > 0:
            return self.xy[self.pos, :]
        return np.array([[], []])

    def _has_pos(self):
        return self.xy_pos.size > 0

    def _num_pos(self):
        if self._has_pos():
            return 0

        return self.pos_candidates.shape[0]

    def _build_output_from_negs(self, idx_negs):
        """
        Builds labels, features, and coordinates arrays
        """
        Y = np.zeros(idx_negs.size).astype(int)
        f = self.descs[idx_negs, :]
        xy = self.xy[idx_negs, :]

        if self._has_pos():
            Y = np.concatenate(
                (np.ones(self.pos_candidates.shape[0]).astype(int), Y))
            f = np.concatenate((self.descs[self.pos_candidates], f))
            xy = np.concatenate((self.xy_pos, xy))

        return Y, f, xy

    def get_n_negs(self, N_neg=None):
        if (self._has_pos()) and (N_neg is None):
            N_neg = self.xy_pos.shape[0]
        elif (not self._has_pos()):
            N_neg = self.default_N_negs

        return N_neg

    def sample(self, N_neg=None, p=None):
        """
        Samples negatives.
        When N_neg is None, return balanced set
        p are the foreground probabilities

        Returns Y, f, xy
            Y: ndarray with labels in {0;1}
            f: ndarray with each row a feature vector
            xy: ndarray with locations of patches
        """
        N_neg = self.get_n_negs(N_neg)

        if p is not None:
            idx_negs = np.argsort(p)[-N_neg:]
        else:
            idx_negs = np.random.choice(self.neg_candidates, size=N_neg)

        return self._build_output_from_negs(idx_negs)


if __name__ == "__main__":

    xy_pos = utls.read_labels('../dataset/labels/0.txt')
    xy, descs = utls.read_feats('../feats/0.npz')
    im = io.imread('../dataset/images/0.png')

    sampler = Sampler(xy_pos, xy, descs)
    Y, f, xy = sampler.sample()

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    utls.plot_bboxes(xy[Y == 1, :], im.shape, ax, color='g')
    utls.plot_bboxes(xy[Y == 0, :], im.shape, ax, color='r')

    plt.show()
