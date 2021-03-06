#!/usr/bin/env python3

import configargparse
import matplotlib.pyplot as plt
import utils as utls
import os
import pickle
import numpy as np
import glob
import pandas as pd
"""
Extract parameters that minimize count error
"""

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--scores-path', default='../results/crossval_scores.p')
    cfg = p.parse_args()

    with open(cfg.scores_path, 'rb') as f:
        scores = pickle.load(f)

    df = pd.DataFrame(scores)
    df = df.groupby(['min_samp_split', 'tau',
                     'max_feats']).mean().reset_index()
    opt_idx = np.argmin(df['mare'])
    opt = df.iloc[opt_idx]

    print('optimal mean absolute relative error: ', opt['mare'])
    print('max_feats: ', opt['max_feats'])
    print('min_samp_split: ', opt['min_samp_split'])
    print('tau: ', opt['tau'])
