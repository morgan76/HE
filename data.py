import random
import numpy as np
import torch
from librosa.util import find_files
import os
from features import get_features
import input_output as io



class CondTripletDset(torch.utils.data.Dataset):

    
    def __init__(self, experiment_config, split):
        
        self.ds_path = experiment_config.ds_path
        self.feat_id = experiment_config.feat_id  
        self.split = split
        self.experiment_config = experiment_config
        self.tracklist = find_files(os.path.join(self.ds_path, 'audio'), ext=self.experiment_config.dataset.audio_exts)

        if self.split == 'valid':
            d_path = '../msaf_/datasets/BeatlesTUT'
            self.tracklist = find_files(os.path.join(d_path, 'audio'), ext=self.experiment_config.dataset.audio_exts)[:10]

        if self.experiment_config.n_conditions == 1:
            self.triplets = self.buildTriplets(self.tracklist, len(self.tracklist), self.experiment_config.batch_size)
        else:
            self.triplets = self.buildTriplets(self.tracklist, len(self.tracklist), 10)
        print('Number of tracks/triplets for split', split, ':', len(self.tracklist), len(self.triplets))


    

    def buildTriplets(self, tracklist, n_songs, n_per_song):
        
        random.seed(None)
        triplets = []

        tracks = np.random.choice(tracklist, size=n_songs, replace=False)

        for track in tracks:
            file_struct = io.FileStruct(track)
            beat_frames = io.read_beats(file_struct.beat_file)
            nb_embeddings = len(beat_frames)
            triplets_temp = []
            
            if nb_embeddings > 128:

                if self.experiment_config.n_conditions == 1:
                    cs = [0]
                elif self.experiment_config.n_conditions == 4:
                    cs = [0, 1, 2, 3] * 3

                anchor_indexes = np.random.choice(np.arange(0, nb_embeddings, 1), size=n_per_song, replace=False)

                for i in range(len(anchor_indexes)):
                    anchor_index = anchor_indexes[i]
                    for k in range(len(cs)):
                        c = cs[k]
                        if c == 0:
                            positive_index, negative_index = self.sampler(anchor_index, nb_embeddings, c)
                            negative_temp = positive_index
                        else:
                            positive_index, negative_index = self.sampler(anchor_index, nb_embeddings, c)
                            negative_index = negative_temp
                            negative_temp = positive_index

                        assert positive_index != negative_index
                        anchor_index_= beat_frames[anchor_index]
                        positive_index_ = beat_frames[positive_index]
                        negative_index_ = beat_frames[negative_index]
                            
                        triplets_temp.append((track, c, anchor_index_, positive_index_, negative_index_))
    
                if self.experiment_config.n_conditions == 1:
                    assert len(triplets_temp) == n_per_song
                else:
                    assert len(triplets_temp) == self.experiment_config.batch_size

                random.shuffle(triplets_temp)
                triplets+=triplets_temp

        return triplets




    def sampler(self, anchor_index, nb_embeddings, c, delta_p = 16, delta_n_min = 1, delta_n_max = 96):
        # Randomly samples positive and negative anchors from beat frames
        L = int(nb_embeddings)

        if self.experiment_config.n_conditions == 1:
            delta_p_min = 1
            delta_p = 16
            delta_n_min = 1
            delta_n_max = 128


        elif self.experiment_config.n_conditions == 4:
            if c == 0:
                delta_p_min = 48
                delta_p = 64
                delta_n_min = 64
                delta_n_max = 128
            elif c == 1:
               delta_p_min = 32
               delta_p = 48
               delta_n_min = 40
               delta_n_max = 48
            elif c == 2:
               delta_p_min = 16
               delta_p = 32
               delta_n_min = 32
               delta_n_max = 40
            elif c == 3:
               delta_p_min = 1
               delta_p = 16
               delta_n_min = 24
               delta_n_max = 32
        

        total_positive = list(np.arange(max(anchor_index-delta_p, 0), max(anchor_index-delta_p_min, 0))) + list(np.arange(min(anchor_index+delta_p_min, L-1), min(anchor_index+delta_p, L-1)))
        positive_index = random.choice(total_positive)
        

        total_negative = list(np.arange(max(anchor_index-delta_n_max, 0), max(anchor_index-delta_n_min, 0))) + list(np.arange(min(anchor_index+delta_n_min, L-1), min(anchor_index+delta_n_max, L-1)))
        if self.experiment_config.n_conditions == 1:
            total_negative.remove(positive_index)
        negative_index = random.choice(total_negative)
        
        return positive_index, negative_index



    def __getitem__(self, index):
        track, c, anchor_index_, positive_index_, negative_index_ = self.triplets[index]
        features = get_features(track, self.feat_id, self.experiment_config)

        features = (features-np.min(features))/(np.max(features)-np.min(features))
        features_padded = np.pad(features,
                    pad_width=((0, 0),
                                (int(self.experiment_config.embedding.n_embedding//2),
                                int(self.experiment_config.embedding.n_embedding//2))
                                ),
                    mode='edge')

        anchor_patch = torch.tensor(features_padded[:, anchor_index_:anchor_index_+self.experiment_config.embedding.n_embedding])[None, :, :]
        positive_patch = torch.tensor(features_padded[:, positive_index_:positive_index_+self.experiment_config.embedding.n_embedding])[None, :, :]
        negative_patch = torch.tensor(features_padded[:, negative_index_:negative_index_+self.experiment_config.embedding.n_embedding])[None, :, :]
        
        return anchor_patch, negative_patch, positive_patch, c


    def __len__(self):
        return len(self.triplets)
