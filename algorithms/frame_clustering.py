import logging
import numpy as np
import scipy.cluster.vq as vq
from sklearn import mixture
from sklearn.cluster import KMeans, AgglomerativeClustering
import librosa
import six
from scipy.ndimage import filters
import scipy

def lognormalize(F, floor=0.1, min_db=-80):
    """Log-normalizes features such that each vector is between min_db to 0."""
    assert min_db < 0
    F = min_max_normalize(F, floor=floor)
    F = np.abs(min_db) * np.log10(F)  # Normalize from min_db to 0
    return F


def min_max_normalize(F, floor=0.001):
    """Normalizes features such that each vector is between floor to 1."""
    F += -F.min() + floor
    F = F / F.max(axis=0)
    return F


def normalize(X, norm_type, floor=0.001, min_db=-80):
    """Normalizes the given matrix of features.

    Parameters
    ----------
    X: np.array
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - `"min_max"`: Min/max scaling is performed
        - `"log"`: Logarithmic scaling is performed
        - `np.inf`: Maximum absolute value
        - `-np.inf`: Mininum absolute value
        - `0`: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed

    Returns
    -------
    norm_X: np.array
        Normalized `X` according the the input parameters.
    """
    if isinstance(norm_type, six.string_types):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)

def get_feat_segments(F, bound_idxs):
    """Returns a set of segments defined by the bound_idxs.

    Parameters
    ----------
    F: np.ndarray
        Matrix containing the features, one feature vector per row.
    bound_idxs: np.ndarray
        Array with boundary indeces.

    Returns
    -------
    feat_segments: list
        List of segments, one for each boundary interval.
    """
    # Make sure bound_idxs are not empty
    assert len(bound_idxs) > 0, "Boundaries can't be empty"

    # Make sure that boundaries are sorted
    bound_idxs = np.sort(bound_idxs)

    # Make sure we're not out of bounds
    assert bound_idxs[0] >= 0 and bound_idxs[-1] < F.shape[0], \
        "Boundaries are not correct for the given feature dimensions."

    # Obtain the segments
    feat_segments = []
    for i in range(len(bound_idxs) - 1):
        feat_segments.append(F[bound_idxs[i]:bound_idxs[i + 1], :])
    return feat_segments


def compute_similarity(F, bound_idxs, k=6):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(F)
    return kmeans.labels_


def post_process(est_labels, in_bound_idxs):
    final_labels = []
    for i in range(1,len(in_bound_idxs)):
        subset = np.array(est_labels[in_bound_idxs[i-1]:in_bound_idxs[i]])
        unique, counts = np.unique(subset, return_counts=True)
        main_label = unique[np.argmax(counts)]
        final_labels.append(main_label)
    return final_labels


def get_boundaries(labels):
    boundaries = []
    clusters = []
    boundaries.append(0)
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            boundaries.append(i)
            clusters.append(labels[i])
    return boundaries, clusters


def frame_cluster(F, in_bound_idxs):
    F = normalize(F, norm_type="min_max")
    est_labels = compute_similarity(F, in_bound_idxs, k=6)
    est_labels = post_process(est_labels, in_bound_idxs)
    return in_bound_idxs, np.array(est_labels)


def get_clusters(F, bounds):
    bounds = list(bounds)
    if 0 not in bounds:
        bounds.insert(0,0)
    bounds.append(F.shape[1]-1)
    clusters = np.arange(len(bounds)-1)
    return list(clusters), bounds


def frame_cluster_bad(F):
    list_boundaries = []
    list_clusters = []
    results = {}
    F = normalize(F, norm_type="min_max")
    for nb_clusters in range(2, 5):
        #print('K1 =', nb_clusters)
        results[nb_clusters] = {}
        try:
            vect_rep = []
            bounds = librosa.segment.agglomerative(F, nb_clusters)
            for i in range(1, len(bounds)):
                vect_rep.append(np.mean(F[:, bounds[i-1]:bounds[i]], axis = 1))
            #for k_2 in range(2, nb_clusters-1):
            #k_2 = int(0.5*nb_clusters)
            if nb_clusters < 8:
                k_2 = int(0.5*nb_clusters)
            else:
                k_2 = 7
            clustering = AgglomerativeClustering(n_clusters=k_2).fit(np.array(vect_rep))
            clusters = clustering.labels_
            #clusters, bounds = get_clusters(F, bounds)
            #print('Boundaries =', bounds)
            #print('Labels =', clusters)
            list_boundaries.append(bounds)
            list_clusters.append(list(clusters))
        except: 
            break
    return list_boundaries, list_clusters


def median_filter(Y, M=8):
    X = np.copy(Y)
    #print(X.shape)
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def frame_cluster_h__(F):
    list_boundaries, list_clusters, results = [], [], {}
    #F = normalize(F, norm_type="min_max")
    #F = scipy.ndimage.median_filter(F, size=(1, 8))
    #F = F.T
    #F = median_filter(F, M=32)
    #nb_c = [3, 7, 11]
    for nb_clusters in range(2, 15):
    #for nb_clusters in nb_c:
        try:
            results[nb_clusters] = {}
            #
            #boundaries = librosa.segment.agglomerative(F, k=nb_clusters)
            clustering = AgglomerativeClustering(n_clusters=nb_clusters).fit(F)
            boundaries = np.where(np.diff(clustering.labels_))[0]
            boundaries = np.insert(boundaries, 0, 0)
            labels = clustering.labels_[boundaries+1]
    
            boundaries = list(boundaries)
            boundaries.append(len(F)-1)
            list_boundaries.append(list(boundaries))
            list_clusters.append(list(labels))
        except:
            break
    return list_boundaries, list_clusters



def frame_cluster_h(F, n_conditions):
    mask_array = np.zeros([n_conditions, 128])
    mask_len = int(128 / n_conditions)
    for i in range(n_conditions):
        mask_array[i, :(i+1)*mask_len] = 1
    
    list_boundaries, list_clusters, results = [], [], {}
    nb_clusters = [2, 3, 4, 5]
    for i in range(n_conditions):
        embedding_temp = F*mask_array[i]
        clustering = KMeans(n_clusters=nb_clusters[i], random_state=0).fit(embedding_temp)
        #clustering = AgglomerativeClustering(n_clusters=nb_clusters).fit(F)
        boundaries = np.where(np.diff(clustering.labels_))[0]
        boundaries = np.insert(boundaries, 0, 0)
        labels = clustering.labels_[boundaries+1]
        boundaries = list(boundaries)
        boundaries.append(len(F)-1)
        list_boundaries.append(list(boundaries))
        list_clusters.append(list(labels))

    return list_boundaries, list_clusters










def frame_cluster_h_(F):
    list_boundaries, list_clusters, results = [], [], {}
    F = normalize(F, norm_type="min_max")
    nb_clusters = [6, 11]
    n_conditions = 2
    mask_lengths = []
    
    #for i in range(1, n_conditions+1):
    #    mask_lengths.append(i * int(128 / n_conditions))
    #mask_lengths[-1] = 128
    mask_lengths = [64, 128]

    for i in range(len(mask_lengths)):
        results[i] = {}
        clustering = AgglomerativeClustering(n_clusters=nb_clusters[i]).fit(F[:,i*64:(i+1)*64])
        #print(clustering.labels_)
        boundaries = np.where(np.diff(clustering.labels_)!=0)[0]
        boundaries = np.insert(boundaries, 0, 0)
        labels = clustering.labels_[boundaries+1]
        boundaries = list(boundaries)
        boundaries.append(len(F)-1)
        list_boundaries.append(list(boundaries))
        list_clusters.append(list(labels))


    return list_boundaries, list_clusters