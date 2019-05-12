import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import argutils
from mano_train.options import expopts, simulopts
from mano_train.netscripts import savemano, simulate
from mano_train.objectutils.objectio import load_obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation')
    expopts.add_exp_opts(parser)
    simulopts.add_simul_opts(parser)
    args = parser.parse_args()
    argutils.print_args(args)
    simulate.full_simul(
        exp_id=args.exp_id,
        batch_step=args.batch_step,
        wait_time=args.wait_time,
        sample_vis_freq=args.sample_vis_freq,
        use_gui=args.use_gui,
        sample_step=args.sample_step,
        workers=args.workers,
        cluster=args.cluster)
    print('All done!')
