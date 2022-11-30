# -*- coding: utf-8 -*-
import argparse

from configuration import config
from train import train_model

import warnings;
warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exp_csn_4levels')
   
    parser.add_argument('feat_id', type=str,
                        help='Feature id.')
    parser.add_argument("ds_path", help="Path to the dataset", type=str,
                        default='.')
    args, _ = parser.parse_known_args()

    # Modify things here
    config.model_name = 'csn_4_levels'
    config.feat_id = args.feat_id
    config.ds_path = args.ds_path
    config.n_conditions = 4
    train_model(config)
