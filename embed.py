# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from features import get_features
from input_output import FileStruct
import input_output as io



# frame-wise embeddings
def embed(embedding_net, frames, device, config):
    batch_size = config.val_batch_size
    with torch.no_grad():
        embedding_net.to(device)
        embedding_net.eval()
        frames = frames.to(device)  
        embeddings = torch.empty((len(frames), config.output_dim))
        idx = 0
        while (idx * batch_size) < len(frames):
            batch = frames[idx * batch_size : (idx + 1) * batch_size]
            embeddings[idx * batch_size : (idx + 1) * batch_size, :] = embedding_net(batch)
            idx += 1
        del frames
        return embeddings.detach().numpy()



def eval_track(audio_file, embedding_net, config, device, feat_id):
    feat_id = config.feat_id

    features = get_features(audio_file, feat_id, config)
    features = (features-np.min(features))/(np.max(features)-np.min(features))
    features_padded = np.pad(features,
                pad_width=((0, 0),
                            (int(config.embedding.n_embedding//2),
                            int(config.embedding.n_embedding//2))
                            ),
                mode='edge')
            
    beat_frames = io.read_beats(FileStruct(audio_file).beat_file)
    features = np.stack([features_padded[:, i:i+config.embedding.n_embedding] for i in beat_frames], axis=0)
    features = torch.tensor(features[:,None, :,:])

    embeddings = embed(embedding_net.embeddingnet.embeddingnet, features, device, config) 
               
    return embeddings


