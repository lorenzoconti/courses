import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# load the wikipedia dataset
wiki = pd.read_csv('people_wiki.csv')


# extract tf-idf features
def load_sparse_csr(filename):
    # load arrays or pickled objects from .npy, .npz or pickled files
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    # compressed sparse row matrix
    # [data] contains all the non zero elements of the sparse matrix
    # [indices] is an array mapping each element in [data] to its column in the sparse matrix
    # [indptr] then maps the elements of [data] and [indices] to the rows of the sparse matrix.
    # if the sparse matrix has M rows, [indptr] is an array containting M+1 elements.
    # for row i, [indptr[i]:indptr[i+1]] returns the indices of elements to take from [data] and [indices] corresponding
    # to row i.
    # so suppose indptr[i]=k, indptr[i+1]=k, the sdata corresponding to row i would be data[k:l] ad columns indices[k:l]
    return csr_matrix((data, indices, indptr), shape)


tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = pd.read_json('people_wiki_map_index_to_word.json', typ='series')

# normalize to be consistent with the k-means assignment
tf_idf = normalize(tf_idf)


def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    """
    cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster
    """
    data_matrix = cluster['matrix']
    dataframe = cluster['dataframe']

    # run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, verbose=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

    # divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child = data_matrix[cluster_assignment == 0]
    data_matrix_right_child = data_matrix[cluster_assignment == 1]

    # divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = np.array(cluster_assignment)  # minor format conversion
    dataframe_left_child = dataframe[cluster_assignment_sa == 0]
    dataframe_right_child = dataframe[cluster_assignment_sa == 1]

    # package relevant variables for the child clusters
    cluster_left_child = {'matrix': data_matrix_left_child,
                          'dataframe': dataframe_left_child,
                          'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}

    return cluster_left_child, cluster_right_child


wiki_data = {'matrix': tf_idf, 'dataframe': wiki}
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=6, seed=1)


def display_single_tf_idf_cluster(cluster, map_index_to_word):
    """map_index_to_word: SFrame specifying the mapping betweeen words and column indices"""

    wiki_subset = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid = cluster['centroid']

    # print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in range(5):
        print('{0:s}:{1:.3f}'.format(map_index_to_word.index[idx[i]], centroid[idx[i]])),
    print('')

    # compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()

    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()

    # for 8 nearest neighbors, print the title as well as first 180 characters of text.
    # wrap the text at 80-character mark.
    for i in range(8):
        text = ' '.join(wiki_subset.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset.iloc[nearest_neighbors[i]]['name'],
                                                           distances[nearest_neighbors[i]], text[:90],
                                                           text[90:180] if len(text) > 90 else ''))
    print('')


display_single_tf_idf_cluster(left_child, map_index_to_word)
display_single_tf_idf_cluster(right_child, map_index_to_word)
