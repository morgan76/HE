# -*- coding: utf-8 -*-
import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description='Neural Features trainer')

    # Dataset parameters
    parser.add_argument('model_name', type=str,
                        help='Name of the model. If it exists, resume'
                        ' training.')
    parser.add_argument('ds_path', type=str,
                        help='Path to the  dataset.')
    parser.add_argument('--bnd-shift', type=float)
    parser.add_argument('--exclude-silence', action='store_false',
                        help='Should silent segments be excluded?',
                        default=None)

    # Files parameters
    parser.add_argument('--checkpoints_dir', type=Path,
                        help="Folder where to store checkpoints.")
    parser.add_argument('--models_dir', type=Path,
                        help="Folers where to store trained models.")

    # Model parameters
    # parser.add_argument('--use-batch-norm', action='store_true')  #TODO: FIX that in parser
    # TODO: add kernel size in parser

    # Feature parameters
    parser.add_argument('--feat-id', type=str,
                        help='Name of the feature to train embedding on')
    parser.add_argument('--feat-type', type=str,
                        help='Name of the method used to compute features')
    parser.add_argument('--sr', type=int, help="Sample rate of input audio.")
    parser.add_argument('--n-fft', type=int)
    parser.add_argument('--hop-size', type=int)
    parser.add_argument('--n-embedding', type=int,
                        help="Number of feature frames in the embedding.")

    # Model parameters
    parser.add_argument('--nb-workers', type=int,
                        help='Number of workers for dataloader.')
    parser.add_argument('--margin', type=float,
                        help='margin parameter of the triplet loss function')
    parser.add_argument('-p', type=int,
                        help='Norm degree for pairwise distance of triplet'
                        ' loss')

    # Training parameters
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('--lr', type=float, dest='learning_rate',
                        help='learning rate, defaults to 1e-2')
    parser.add_argument('--mining', type=str,
                        help='Strategy for triplet mining')
    parser.add_argument('--m-per-class', type=int,
                        help="Number of samples per class in each batch")
    parser.add_argument('--classes', type=str, dest='classes_strategy',
                        choices=('track', 'label'))
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true', default=None,
                        help='disables CUDA training')
    parser.add_argument('--quiet', action='store_true', default=None,
                        help='less verbose during training')
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Parameter to resume training of a model.")

    return parser
