import librosa
import scipy
import scipy.linalg as linalg
import scipy.signal as sig
import numpy as np
import sklearn.cluster as sklhc
import scipy.cluster.hierarchy as scihc
from collections import OrderedDict
from .oracles.utils import entropy


"""Segmentation algorithms
"""
def _seg_by_hc_single_frame(obs_len, connectivity, data, width=9, hier=False, **kwargs):
    _children, _n_c, _n_leaves, parents, distances = \
        sklhc.ward_tree(data, connectivity=connectivity, return_distance=True)

    reconstructed_z = np.zeros((obs_len - 1, 4))
    reconstructed_z[:, :2] = _children
    reconstructed_z[:, 2] = distances

    if 'criterion' in kwargs.keys():
        criterion = kwargs['criterion']
    else:
        criterion = 'distance'

    if hier:
        t_list = range(2, 11)

        label_dict = OrderedDict()
        boundary_dict = OrderedDict()
        criterion = 'maxclust'
        for t in t_list:
            boundaries, labels = _agg_segment(reconstructed_z, t, criterion, width, data)
            label_dict[np.max(labels) + 1] = labels
            boundary_dict[np.max(labels) + 1] = boundaries
        return boundary_dict, label_dict
    else:
        t = 0.7 * np.max(reconstructed_z[:, 2])
        return _agg_segment(reconstructed_z, t, criterion, width, data)

def segment_labeling(x, boundaries, c_method='kmeans', k=5):
    x_sync = librosa.util.utils.sync(x.T, boundaries[:-1])
    if c_method == 'kmeans':
        c = sklhc.KMeans(n_clusters=k, n_init=100)
        seg_labels = c.fit_predict(x_sync.T)
    elif c_method == 'agglomerative':
        z = scihc.linkage(x_sync.T, method='ward')
        t = k * np.max(z[:, 2])
        seg_labels = scihc.fcluster(z, t=t, criterion='distance')
    else:
        c = sklhc.KMeans(n_clusters=k, n_init=100)
        seg_labels = c.fit_predict(x_sync.T)

    return seg_labels

def _agg_segment(z, t, criterion, width, data):
    label = scihc.fcluster(z, t=t, criterion=criterion)
    k = len(np.unique(label))
    boundaries = find_boundaries(label, width=width)
    while len(boundaries) < k + 1 and width > 0:
        width -= 3
        boundaries = find_boundaries(label, width=width - 3)
    labels = segment_labeling(data, boundaries, c_method='kmeans', k=k)
    return boundaries, labels

def _seg_by_single_frame(oracle, cluster_method='agglomerative', connectivity='temporal', data='symbol',
                         median_filter_width=9, **kwargs):
    obs_len = oracle.statistics['n_states'] - 1
    median_filter_width = median_filter_width

    if data == 'raw':
        data = np.array(oracle.f_array[1:])
    else:
        data = np.zeros((oracle.statistics['n_states'] - 1, oracle.num_clusters()))
        data[range(oracle.statistics['n_states'] - 1), oracle.basic_attributes['data'][1:]] = 1

    if connectivity == 'temporal':
        connectivity = np.zeros((obs_len, obs_len))

    if cluster_method == 'agglomerative':
        return _seg_by_hc_single_frame(obs_len=obs_len, connectivity=connectivity, data=data, **kwargs)

def segmentation(oracle, method='symbol_agglomerative', **kwargs):
    if oracle:
        if method == 'symbol_agglomerative':
            return _seg_by_single_frame(oracle, cluster_method='agglomerative', **kwargs)

    else:
        raise TypeError('Oracle is None')


def find_boundaries(frame_labels, width=9):
    # frame_labels = np.pad(frame_labels, (int(width / 2), int(width / 2) + 1), mode='edge')
    # frame_labels = np.array([stats.mode(frame_labels[i:j])[0][0]
    #                          for (i, j) in zip(range(0, len(frame_labels) - width),
    #                                            range(width, len(frame_labels)))])
    boundaries = 1 + np.flatnonzero(frame_labels[:-1] != frame_labels[1:])
    # np.asarray(np.where(frame_labels[:-1] != frame_labels[1:])).reshape((-1,))
    boundaries = np.unique(np.concatenate(
        [[0], boundaries, [len(frame_labels) - 1]]))
    return boundaries


def normalized_graph_laplacian(mat):
    mat_inv = 1. / np.sum(mat, axis=1)
    mat_inv[~np.isfinite(mat_inv)] = 1.
    mat_inv = np.diag(mat_inv ** 0.5)
    laplacian = np.eye(len(mat)) - mat_inv.dot(mat.dot(mat_inv))

    return laplacian


def eigen_decomposition(mat, k=6):  # Changed from 11 to 8 then to 6(7/22)
    vals, vecs = linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    idx = np.argsort(vals)

    vals = vals[idx]
    vecs = vecs[:, idx]

    if len(vals) < k + 1:
        k = -1
    vecs = scipy.ndimage.median_filter(vecs, size=(5, 1))
    return vecs[:, :k]
