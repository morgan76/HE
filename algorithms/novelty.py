import numpy as np
import scipy
import pandas as pd
import librosa

def F_tests(ref_labels, ref_times, embeddings, beat_times):
    M = 1
    # for each boundary time
    # define neighboring window of size M (before and after)
    # build first-order difference function
    # find peak within close neighborhood M' of the anchor (must be bigger than the rest over M)
    #ref_frames = librosa.time_to_frames(ref_times, sr=22050, hop_length=64)
    
    _, labels, __, results = get_labels(ref_labels, ref_times, beat_times)
    labels_list = list(results.keys())
    print(ref_times, labels_list)
    for i in range(len(labels_list)-1):
        #patch = np.concatenate([embeddings[results[labels_list[i]], :][-M:], embeddings[results[labels_list[i+1]], :][:M]])
        patch = np.concatenate([embeddings[results[labels_list[i]], :], embeddings[results[labels_list[i+1]], :]])
        print('Patch shape =', patch.shape)
        diff = np.zeros_like(patch)
        diff[1:,] = np.diff(patch, axis=0)
        novelty = np.max([np.zeros_like(diff), diff], axis=0)
        novelty_sum = np.sum(novelty, axis=1)
        tol = len(novelty_sum)
        novelty_peaks = librosa.util.peak_pick(novelty_sum, pre_max=tol, post_max=tol, pre_avg=tol, post_avg=tol, 
                               delta=0, wait=0)
        #print(results[labels_list[i]][-1]+novelty_peaks[0])
        #print('Peak found =',librosa.frames_to_time((results[labels_list[i]][-M]+novelty_peaks)*128, sr=22050, hop_length=64))
        #peak = beat_times[results[labels_list[i]][-M]+novelty_peaks[0]]
        peak = beat_times[results[labels_list[i]][0]+novelty_peaks[0]]
        print('Peak found =',peak)
        #start = librosa.frames_to_time(results[labels_list[i]][0]*128, sr=22050, hop_length=64)
        #start = beat_times[results[labels_list[i]][-M]]
        start = beat_times[results[labels_list[i]][0]]
        #end = librosa.frames_to_time(results[labels_list[i+1]][-1]*128, sr=22050, hop_length=64)
        #end = beat_times[results[labels_list[i+1]][M]]
        end = beat_times[results[labels_list[i+1]][-1]]
        print('Start of patch =', start)
        print('End of patch =', end)
        print('True boundary =',ref_times[i+1])
        print('-------------')
        #patch = np.c_[embeddings[results[labels_list[i]], :], embeddings[results[labels_list[i+1]], :]]
        #print(patch.shape)
    return 0



def get_labels(ref_labels, ref_times, inter_times):
    labels = []
    res_labels = {}
    res_ind_labels = {}
    ind_labels_list = []
    ind_labels = np.arange(0,len(ref_labels),1)
    bnd_shift = 0
    # iterate over beat indexes
    for i in range(len(inter_times)):
        # iterate over boundaries
        for j in range(1,len(ref_times)):
            if inter_times[i] > ref_times[j-1]+bnd_shift and inter_times[i] < ref_times[j]-bnd_shift:
                # keep the label in memory 
                labels.append(ref_labels[j-1])
                # keep the one from overall list
                ind_labels_list.append(ind_labels[j-1])

                if ref_labels[j-1] in res_labels.keys():
                    res_labels[ref_labels[j-1]].append(i)
                else:
                    res_labels[ref_labels[j-1]] = [i]

                if ind_labels[j-1] in res_ind_labels.keys():
                    res_ind_labels[ind_labels[j-1]].append(i)
                else:
                    res_ind_labels[ind_labels[j-1]] = [i]
                break
    #print(len(labels))
    #print(len(ind_labels_list))
    return labels, ind_labels_list, res_labels, res_ind_labels


