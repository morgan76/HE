import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing

def F_tests(ref_labels, ref_times, embeddings, inter_times):
    #embeddings = (embeddings-np.min(embeddings))/(np.max(embeddings)-np.min(embeddings))
    #embeddings = embeddings.T
    res_ind = 0
    F_ind = 0
    le = preprocessing.LabelEncoder()
   # print('shape features = ',embeddings.shape)
    var_tot = np.sum(np.var(embeddings, axis=0))*embeddings.shape[0]
    #print('shape variance =',np.var(embeddings, axis=0).shape)
    #print('Var tot =', var_tot)

    # Consider sections independently
    labels_, labels, res_labels, results = get_labels(ref_labels, ref_times, inter_times)
    labels = le.fit_transform(labels)
    for label in results.keys():
        sub_embed = embeddings[results[label], :]
        #print('Dim sub embed =', sub_embed.shape)
        var_sub = np.sum(np.var(sub_embed,axis=0))*sub_embed.shape[0]
        #print('shape variance =',np.var(sub_embed,axis=0).shape)
        F_ind += var_sub
    #print('F_ind =', F_ind)
    res_ind = F_ind/var_tot

    res_labels = 0
    F_labels = 0

    # Use annotated sections:
    labels, ind_labels, results, res_ind_labels = get_labels(ref_labels, ref_times, inter_times)   
    labels = le.fit_transform(labels)
    for label in results.keys():
        sub_embed = embeddings[results[label], :]
        #print('Dim sub embed =', sub_embed.shape)
        var_sub = np.sum(np.var(sub_embed,axis=0))*sub_embed.shape[0]
        #print('shape variance =',np.var(sub_embed,axis=0).shape)
        F_labels += var_sub
    #print('F_label =', F_labels)
    res_labels = F_labels/var_tot


    #print('___________________________________')
    return res_ind, res_labels



def get_labels(ref_labels, ref_times, inter_times):
    labels = []
    res_labels = {}
    res_ind_labels = {}
    ind_labels_list = []
    ind_labels = np.arange(0,len(ref_labels),1)
    bnd_shift = 1
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


