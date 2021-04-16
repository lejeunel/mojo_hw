#!/usr/bin/env python3
import numpy as np
from skimage import feature as ft
from skimage import io
import configargparse
import glob
import os
from tqdm import tqdm


class FeatExtractor:
    def __init__(self,
                 step=1,
                 radius=0.01,
                 orientations=4,
                 histograms=4,
                 rings=3):
        # daisy parameters
        self.step = step
        self.radius = radius
        self.orientations = orientations
        self.histograms = histograms
        self.rings = rings

    def extract(self, im):
        # im: numpy nd array

        if im.ndim > 2:
            im = im[..., 0]

        # pad image to account for DAISY radius
        im_padded = np.pad(im, (int(self.radius * np.max(im.shape)), ),
                           mode='mean')

        descs = ft.daisy(im_padded,
                         step=self.step,
                         radius=int(self.radius * np.max(im.shape)),
                         histograms=self.histograms,
                         orientations=self.orientations,
                         rings=self.rings)

        locs = np.meshgrid(np.linspace(0, 1, descs.shape[1]),
                           np.linspace(0, 1, descs.shape[0]))

        return locs[0].ravel(), locs[1].ravel(), descs


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--in-path', default='../dataset/images')
    p.add('--im-ext', default='png')
    p.add('--out-path', default='../feats')
    p.add('--step', default=16)
    p.add('--histograms', default=8)
    p.add('--radius', default=0.01)
    p.add('--rings', default=3)
    p.add('--orientations', default=8)
    cfg = p.parse_args()

    if not os.path.exists(cfg.out_path):
        print('creating output path ', cfg.out_path)
        os.makedirs(cfg.out_path)

    featex = FeatExtractor(cfg.step,
                           radius=cfg.radius,
                           orientations=cfg.orientations,
                           histograms=cfg.histograms,
                           rings=cfg.rings)

    ims_fname = sorted(glob.glob(os.path.join(cfg.in_path, '*.' + cfg.im_ext)))
    print('found ', len(ims_fname), ' images')

    for im_fname in tqdm(ims_fname):
        base_name = os.path.splitext(os.path.split(im_fname)[-1])[0]

        out_path = os.path.join(cfg.out_path, base_name + '.npz')
        if not os.path.exists(out_path):

            im = io.imread(im_fname, as_gray=True)
            x, y, descs = featex.extract(im)
            np.savez(
                out_path, **{
                    'x':
                    x.ravel(),
                    'y':
                    y.ravel(),
                    'descs':
                    descs.astype(np.float16).reshape((-1, descs.shape[-1]))
                })
