#!/usr/bin/env python3

import configargparse
import matplotlib.pyplot as plt
import utils as utls
import os
import pickle
import numpy as np
import glob

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--scores-path', default='../results/crossval_scores.p')
    cfg = p.parse_args()

    with open(cfg.scores_path, 'rb') as f:
        scores = pickle.load(f)
        import pdb
        pdb.set_trace()  ## DEBUG ##
