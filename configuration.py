# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np


class SubConfig():
    def __init__(self, parameters):
        for parameter_key in parameters:
            setattr(self, parameter_key, parameters[parameter_key])

    def __repr__(self):
        variables = vars(self)
        text = ''
        for var in variables:
            text += '    {:20}{}  \n'.format(var, variables[var])
        return text


class Config():
    def __init__(self, parameters):
        for parameter_key in parameters:
            setattr(self, parameter_key, parameters[parameter_key])

    def add_subconfig(self, subconfig_name, subconfig):
        setattr(self, subconfig_name, subconfig)

    def __repr__(self):
        variables = vars(self)
        text = ''
        for var in variables:
            if type(variables[var]) is SubConfig:
                text += '{}  \n'.format(var)
                text += str(variables[var])
            else:
                text += '{:24}{}  \n'.format(var, variables[var])
        return text

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
default_config_dict = {
    'default_bound_id': 'sf',  # Default boundary detection algorithm ("sf", "cnmf", "foote", "olda", "scluster", "gt")
    'default_label_id': None,  # Default label detection algorithm (None, "cnmf", "fmc2d", "scluster")
    'sample_rate': 22050,  # Default Sample Rate to be used.
    'n_fft': 2048,  # FFT size
    'hop_length': 256,  # Hop length in samples
    }

# Files and dirs
default_config_dict['results_dir'] = "results"
default_config_dict['results_ext'] = ".csv"
default_config_dict['out_boundaries_ext'] = "-bounds.wav"
default_config_dict['minimum_frames'] = 10  # Minimum number of frames to activate
default_config_dict['features_tmp_file'] = ".features_msaf_tmp.json"

config = Config(default_config_dict)

# Dataset files and dirs
dataset_subconfig_dict = {
    'audio_dir': "audio",
    'estimations_dir': "estimations",
    'features_dir': "features",
    'references_dir': "references",
    'audio_exts': ['wav', 'mp3', 'aiff'],
    'estimations_ext': ".jams",
    'features_ext': ".json",
    'references_ext': ".jams"
}
dataset_subconfig = SubConfig(dataset_subconfig_dict)
config.add_subconfig('dataset', dataset_subconfig)

# Spectrogram
spectrogram_config_dict = {
    'ref_power': 'max'}
spectrogram_config = SubConfig(spectrogram_config_dict)
config.add_subconfig('spectrogram', spectrogram_config)

# Constant-Q transform
cqt_config_dict = {
    'bins': 120,
    'norm': np.inf,
    'filter_scale': 1.0,
    'ref_power': 'max',
    }
cqt_config = SubConfig(cqt_config_dict)
config.add_subconfig('cqt', cqt_config)

# Melspectrogram
mel_config_dict = {
    'n_mels': 60,  # Number of mel filters
    'ref_power': 'max',  # Reference function to use for the logarithm power.
    'fmax': 8000
    }


mel_config = SubConfig(mel_config_dict)
config.add_subconfig('mel', mel_config)


mfcc_config_dict = {
    'n_mels': 60,  # Number of mel filters
    'ref_power': 'max',  # Reference function to use for the logarithm power.
    'fmax': 8000
    }


mfcc_config = SubConfig(mfcc_config_dict)
config.add_subconfig('mfcc', mfcc_config)

# Chomagram
chromagram_config_dict = {
    'norm': np.inf,
    'ref_power' : 'max'
    }

harmonic_config_dict = {}
harmonic_config = SubConfig(harmonic_config_dict)
config.add_subconfig('harmonic', harmonic_config)

chromagram_config = SubConfig(chromagram_config_dict)
config.add_subconfig('chroma', chromagram_config)

# Embed features
embedding_feat_config_dict = {'base_feat_id': 'mel',
                              'n_embedding': 512,  # Number of feature frames in the learned embedding,
                              'embed_hop_length': 86  # Embedding hop length in feature frames
                              }
embedding_config = SubConfig(embedding_feat_config_dict)
config.add_subconfig('embedding', embedding_config)

# experiment_config = config

# Dataset parameters
config.model_name = None
config.ds_path = None

# Files parameters
config.checkpoints_dir = Path('results').joinpath('checkpoints')
config.models_dir = Path('results').joinpath('models')
config.tensorboard_logs_dir = Path('results').joinpath('runs')

# Model parameters
config.use_batch_norm = False
config.architecture = 'FC'
config.prein = True #equally-spaced disjoint masks
config.learnedmask = False #learn the masks
config.n_conditions = 4 #1 for flat embeddings, 4 for multi-level embeddings
config.use_dropout = True 
config.output_dim = 128 #output embedding dimension

# Default feature parameters
config.feat_id = 'mel'

# Training parameters
config.epochs = 250
config.batch_size = 120
config.val_batch_size = 120
config.learning_rate = 1e-5
config.resume = False
config.no_cuda = False
config.quiet = False
config.nb_workers = 0
config.seed = 42
config.patience = 10

# Validation parameters
config.annot_level = 0 
config.annotator_id = 0