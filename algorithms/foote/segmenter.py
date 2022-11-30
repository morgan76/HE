# coding: utf-8
"""Copied (almost completely) from MSAF:
https://github.com/urinieto/msaf/blob/master/msaf/input_output.py
"""
import numpy as np
from scipy.spatial import distance
from scipy import signal
#from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import scipy
import librosa
import utils


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    
    #for i in range(X.shape[0]):
    #    X[i, :] = filters.median_filter(X[i, :], size=M)
    return X

def median_filter_(X, M):
    """Median filter along the first axis of the feature matrix X."""
    return signal.medfilt2d(X, kernel_size=M)


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.gaussian(M, M//3., sym=True)
    #g = signal.gaussian(M, 18, sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
    G[:M // 2, M // 2:] = -G[:M // 2, M // 2:]
    return G


def compute_ssm(X, metric="sqeuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    #D = librosa.segment.recurrence_matrix(X, mode='affinity', sym=True, k=10)
    D /= D.max()
    #df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    #D = df(D, size=(1, 4))
    return 1 - D
    #return D


        
def compute_nc(X, G):
    """Computes the novelty curve from the self-similarity matrix X and
        the gaussian kernel G."""
    N = X.shape[0]
    M = G.shape[0]
    nc = np.zeros(N)

    for i in range(M // 2, N - M // 2 + 1):
        nc[i] = np.sum(X[i - M // 2:i + M // 2, i - M // 2:i + M // 2] * G)
    # Normalize
    nc += nc.min()
    nc /= nc.max()
    #nc = (nc-nc.min())/(nc.max()-nc.min())
    return nc


def pick_peaks(nc, L):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() / 20.

    nc = filters.gaussian_filter1d(nc, sigma=4)  # Smooth out nc
    th = filters.median_filter(nc, size=L) + offset
    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    return peaks



def pick_peaks_(nc, L, T_back_mean, T_for_mean, T_back_max, T_for_max):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    peaks = []
    nc = filters.gaussian_filter1d(nc, sigma=4)  # Smooth out nc
    #nc = signal.medfilt(nc, kernel_size=31)
    #print('Len NC = ',len(nc))

    # BEATLES
    #peaks = librosa.util.peak_pick(nc, pre_max=T_back_max, post_max=T_for_max, pre_avg=T_back_mean, post_avg=T_for_mean, 
    #                           delta=0.012, wait=0)

    # SALAMI-IA
    peaks = librosa.util.peak_pick(nc, pre_max=T_back_max, post_max=T_for_max, pre_avg=T_back_mean, post_avg=T_for_mean, 
                               delta=0, wait=0)

    
    return peaks
    #return res







def segment(features, T_back_mean, T_for_mean, T_back_max, T_for_max, M_gaussian, m_median, L_peaks,
            bound_norm_feats="min_max"):
    """[summary]

    Args:
        audio_file ([type]): [description]
        feat_id ([type]): [description]
        feat_type ([type]): [description]
        M_gaussian (int, optional): [description]. Defaults to 66.
        m_median (int, optional): [description]. Defaults to 12.
        L_peaks (int, optional): [description]. Defaults to 64.
        bound_norm_feats (str, optional): [description]. Defaults to "min_max".
            "min_max", "log", np.inf, -np.inf, float >= 0, None

    Returns
    -------
    est_idxs : np.array(N)
        Estimated indeces the segment boundaries in frames.
    est_labels : np.array(N-1)
        Estimated labels for the segments.
    """

    # Preprocess to obtain features
    # F = get_features(audio_file, feat_id, config, feat_type).T
    F = features.T
    # Normalize
    F = utils.normalize(F, norm_type="min_max")
    # plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

    # Make sure that the M_gaussian is even
    if M_gaussian % 2 == 1:
        M_gaussian += 1

    # Median filter
    F = median_filter(F, M=m_median)
    # plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

    # Self similarity matrix
    S = compute_ssm(F)
    #S = median_filter(S, M=m_median)
    # Median filter
    #S = median_filter_(S, m_median)
    #plt.imshow(S, interpolation="nearest"); plt.show()

    # Compute gaussian kernel
    G = compute_gaussian_krnl(M_gaussian)
    # plt.imshow(S, interpolation="nearest", aspect="auto"); plt.show()

    # Compute the novelty curve
    nc = compute_nc(S, G)
    #nc = compute_nc_(S)
    
    est_idxs = pick_peaks_(nc, L_peaks, T_back_mean, T_for_mean, T_back_max, T_for_max)
    #est_idxs = pick_peaks(nc, L_peaks)
    # Add first and last frames
    #print(F.shape)
    est_idxs = np.concatenate(([0], est_idxs, [F.shape[0] - 1]))
    
    # Empty labels
    est_labels = np.ones(len(est_idxs) - 1) * -1

    # Post process estimations
    est_idxs, est_labels = utils.postprocess(est_idxs, est_labels)
    
    return est_idxs, est_labels, S, nc  # also return SSM and novelty curve for analysis
    # plt.figure(1)
    # plt.plot(nc);
    # [plt.axvline(p, color="m") for p in est_bounds]
    # [plt.axvline(b, color="g") for b in ann_bounds]
    # plt.figure(2)
    # plt.imshow(S, interpolation="nearest", aspect="auto")
    # [plt.axvline(b, color="g") for b in ann_bounds]
    # plt.show()
