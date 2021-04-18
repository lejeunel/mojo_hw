#!/usr/bin/env python3

import configargparse
import matplotlib.pyplot as plt
import utils as utls
import os
import pickle
import numpy as np
import glob
import pandas as pd

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--scores-path', default='../results/crossval_scores.p')
    p.add('--title', default='')
    p.add('--out-fname', default='cross_validation.png')
    cfg = p.parse_args()

    with open(cfg.scores_path, 'rb') as f:
        scores = pickle.load(f)

    folds = np.unique([s['fold'] for s in scores])
    tau = np.unique([s['tau'] for s in scores])

    df = pd.DataFrame(scores)
    error_mean = df.groupby('tau').mean()['abs_diff']
    error_std = df.groupby('tau').std()['abs_diff']

    plt.plot(tau, error_mean, 'b-')
    plt.plot(tau, error_mean + error_std, 'b--')
    plt.plot(tau, error_mean - error_std, 'b--')
    plt.xlabel('tau')
    plt.ylabel('absolute error')
    plt.grid()
    plt.title(cfg.title)
    plt.tight_layout()
    out_path = os.path.join(os.path.split(cfg.scores_path)[0], cfg.out_fname)
    print('saving figure to ', out_path)
    plt.savefig(out_path)

    print('minimum absolute error: ', np.min(error_mean))
    print('minimum normalized absolute error: ',
          np.min(error_mean) / np.unique(df['count_y']).sum())
    print('tau: ', tau[np.argmin(error_mean)])
