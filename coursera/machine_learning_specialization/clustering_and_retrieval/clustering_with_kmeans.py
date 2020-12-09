import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import sys
import time
import os

# load the data
wiki = pd.read_csv('people_wiki.csv')


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)


# extract tf-idf features
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
map_index_to_word = pd.read_json('people_wiki_map_index_to_word.json', typ='series')

tf_idf = normalize(tf_idf)


# implement k-means
def get_initial_centroids(data, k, seed=None):
    """Randomly choose k data points as initial centroids"""

    if seed is not None:
        np.random.seed(seed)

    # number of data points
    n = data.shape[0]

    # pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    # keep centroids as dense format, as many entries will be nonzero due to averaging.
    # as long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices, :].toarray()

    return centroids


# after initialization, the k-means algorithm iterates between the following two steps:
# 1. assign each data point to the closest centroid
# 2. revise centroids as the mean of the assigned data points

def arg(distances):
    cluster_assignment = list()
    for i in range(0, len(distances)):
        cluster_assignment.append(np.argmin(distances[i]))
    return np.array(cluster_assignment)


# initialize three centroids with the first three rows of tf_idf
centroids = tf_idf[0:3, :]
# compute distances from each of the centroids to all data points in tf_idf
distances = pairwise_distances(tf_idf, centroids)
# use these distance calculations to compute cluster assignments and assign them to cluster_assignment
print(arg(distances))


def assign_clusters(data, centroids):
    # compute distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids)

    # compute cluster assignments for each data point:
    cluster_assignment = arg(distances_from_centroids)

    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []

    for i in range(k):
        # select all data points that belong to cluster i
        member_data_points = data[cluster_assignment == i]
        # compute the mean of the data points
        centroid = member_data_points.mean(axis=0)

        # convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)

    new_centroids = np.array(new_centroids)

    return new_centroids


result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))

# assessing convergence
# how can we tell if the k-means algorithm is converging? we can look at the cluster assignments and see if they
# stabilize over time


def compute_heterogeneity(data, k, centroids, cluster_assignment):

    heterogeneity = 0.0
    for i in range(k):

        # select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[cluster_assignment == i, :]

        # check if i-th cluster is non-empty
        if member_data_points.shape[0] > 0:
            # compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    """This function runs k-means on given data and initial set of centroids.
       maxiter: maximum number of iterations to run.
       record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
       verbose: if True, print how many data points changed their cluster labels in each iteration"""

    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(maxiter):
        if verbose:
            print(itr)

        # 1. make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)

        # 2. compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)

        # check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and (prev_cluster_assignment == cluster_assignment).all():
            break

        # print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))

        # record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


# plotting convergence metric across iterations
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7, 4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400, record_heterogeneity=heterogeneity)
plot_heterogeneity(heterogeneity, k)

print(np.bincount(cluster_assignment))

# beware of local maxima

k = 10
heterogeneity = {}

start = time.time()
quiz_ans = list()

for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:

    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400, record_heterogeneity=None)

    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    print(max(np.bincount(cluster_assignment)))
    quiz_ans.append(max(np.bincount(cluster_assignment)))
    sys.stdout.flush()

end = time.time()
print(end-start)

print(min(quiz_ans), max(quiz_ans))


def smart_initialize(data, k, seed=None):
    """Use k-means++ to initialize a good set of centroids"""
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)

    centroids = np.zeros((k, data.shape[1]))

    # randomly choose the first centroid.
    # since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx, :].toarray()

    # compute distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()

    for i in range(1, k):
        # choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))
        print(idx)
        centroids[i] = data[idx, :].toarray()

        # compute distances from the centroids to all data points
        distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean'), axis=1)

    return centroids


k = 10
heterogeneity_smart = {}
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = smart_initialize(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
    # To save time, compute heterogeneity only once in the end
    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)

plt.figure(figsize=(8, 5))
plt.boxplot([list(heterogeneity.values()), list(heterogeneity_smart.values())], vert=False)
plt.yticks([1, 2], ['k-means', 'k-means++'])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

# random initialization results in a worse clustering than k-means++ on average
# the best result of k-means++ is better than the best result of random initialization


def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    heterogeneity = {}

    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None

    for i in range(num_runs):

        # use UTC time if no seeds are provided
        if seed_list is not None:
            seed = seed_list[i]
            np.random.seed(seed)
        else:
            seed = int(time.time())
            np.random.seed(seed)

        # use k-means++ initialization
        initial_centroids = smart_initialize(data, k, seed)

        # run k-means
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None,
                                               verbose=False)

        # to save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)

        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()

        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    # return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment, best_seed


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


filename = 'kmeans-arrays.npz'
heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}

    for k in k_list:
        print(k)
        sys.stdout.flush()
        '''To save memory space, do not load the arrays from the file right away. We use
           a technique known as lazy evaluation, where some expressions are not evaluated
           until later. Any expression appearing inside a lambda function doesn't get
           evaluated until the function is called.
           Lazy evaluation is extremely important in memory-constrained setting, such as
           an Amazon EC2 t2.micro instance.'''
        centroids[k] = lambda k = k: arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = lambda k=k: arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k](), cluster_assignment[k]())
        heterogeneity_values.append(score)

    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('file not found')

# visualize cluster documents

def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word,
                                display_content=True):
    '''wiki: original dataframe
       tf_idf: data matrix, sparse matrix format
       map_index_to_word: SFrame specifying the mapping betweeen words and column indices
       display_content: if True, display 8 nearest neighbors of each centroid'''

    print('==========================================================')

    # visualize each cluster c
    for c in range(k):
        # Cluster heading
        print('Cluster {0:d}    '.format(c)),

        # Print top 5 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]

        for i in range(5):  # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word.index[idx[i]], centroids[c, idx[i]])),
        print('')

        if display_content:
            # compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[cluster_assignment != c] = float('inf')  # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # for 8 nearest neighbors, print the title as well as first 180 characters of text.
            # wrap the text at 80-character mark.
            for i in range(8):
                text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
                print('\n* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki.iloc[nearest_neighbors[i]]['name'],
                                                                     distances[nearest_neighbors[i]], text[:90],
                                                                     text[90:180] if len(text) > 90 else ''))
        print('==========================================================')


visualize_document_clusters(wiki, tf_idf, centroids[2](), cluster_assignment[2](), 2, map_index_to_word)

