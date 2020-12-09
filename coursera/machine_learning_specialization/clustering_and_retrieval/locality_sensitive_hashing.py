import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt


def norm(x):
    sum_sq = x.dot(x.T)
    norm = np.sqrt(sum_sq)
    return norm


wiki = pd.read_csv('people_wiki.csv')
print(wiki.head())


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


corpus = load_sparse_csr('people_wiki_tf_idf.npz')
assert corpus.shape == (59071, 547979)


# train an LSH model

# LSH performs an efficient neighbor search by randomly partitioning all reference data points into different bins.
# Today we will build a popular variant of LSH known as random binary projection, which approximates cosine distance.
# There are other variants we could use for other choices of distance metrics.
# The first step is to generate a collection of random vectors from the standard Gaussian distribution

def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)


def train_lsh(data, num_vector=16, seed=None):

    dim = data.shape[1]

    if seed is not None:
        np.random.seed(seed)

    random_vectors = generate_random_vectors(num_vector, dim)

    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    table = {}

    # partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # if no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = list()
        # fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model


model = train_lsh(corpus, num_vector=16, seed=143)

# inspect bins
print(wiki[wiki['name'] == 'Barack Obama'])
print(model['bin_indices'][35817])


print(np.array(model['bin_index_bits'][24478], dtype=int))
print(model['bin_indices'][24478])
sum(model['bin_index_bits'][24478] == model['bin_index_bits'][35817])

# There are four other documents that belong to the same bin. Which documents are they?

doc_ids = list(model['table'][model['bin_indices'][35817]])
# display documents other than Obama
doc_ids.remove(35817)

docs = wiki[wiki.index.isin(doc_ids)]
print(docs)


def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]


obama_tf_idf = corpus[35817, :]
biden_tf_idf = corpus[24478, :]

print('================= Cosine distance from Barack Obama')
print('Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf)))
for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id,:]
    print('Barack Obama - {0:24s}: {1:f}'.format(wiki.iloc[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf)))

# It turns out that Joe Biden is much closer to Barack Obama than any of the four documents, even though Biden's bin
# representation differs from Obama's by 2 bits.

# Similar data points will in general tend to fall into nearby bins, but that's all we can say about LSH. In a
# high-dimensional space such as text features, we often get unlucky with our selection of only a few random vectors
# such that dissimilar data points go into the same bin while similar data points fall into different bins. Given a
# query document, we must consider all documents in the nearby bins and sort them according to their actual distances
# from the query.

# query the LSH model

num_vector = 16
search_radius = 3


def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.

    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document

    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    # allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)

    for different_bits in combinations(range(num_vector), search_radius):
        # flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = 1 - alternate_bits[0]

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to the bin indexed by the new bit vector.
        # then add those documents to candidate_set
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set


obama_bin_index = model['bin_index_bits'][35817]
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)


# collect all candidates and compute their true distance to the query
def query(vec, model, k, max_search_radius):
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)

    # Sort candidates by their true distances from the query
    nearest_neighbors = pd.DataFrame({'id': list(candidate_set)})
    candidates = data[np.array(list(candidate_set)), :]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()

    return nearest_neighbors.nsmallest(k, 'distance', )[['id', 'distance']], len(candidate_set)


print(query(corpus[35817, :], model, k=10, max_search_radius=3)[0].set_index('id').join(wiki[['name']], how='inner').
      sort_values('distance'))

# experimenting the LSH implementation

# How does nearby bin search affect the outcome of LSH?
# There are three variables that are affected by the search radius:
# - Number of candidate documents considered
# - Query time
# - Distance of approximate neighbors from the query

num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in range(17):
    start = time.time()
    result, num_candidates = query(corpus[35817, :], model, k=10,
                                   max_search_radius=max_search_radius)
    end = time.time()
    query_time = end - start

    print('Radius:' + str(max_search_radius))

    print(result.set_index('id').join(wiki[['name']], how='inner').sort_values('distance'))

    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()

    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)

plt.figure(figsize=(7, 4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


# quality metrics for neighbors
def brute_force_query(vec, data, k):
    num_data_points = data.shape[0]

    # compute distances for all data points in training set
    nearest_neighbors = pd.DataFrame({'id': range(num_data_points)})
    nearest_neighbors['distance'] = pairwise_distances(data, vec, metric='cosine').flatten()

    return nearest_neighbors.nsmallest(k, 'distance', )



max_radius = 17
precision = {i: [] for i in range(max_radius)}
average_distance = {i: [] for i in range(max_radius)}
query_time = {i: [] for i in range(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('{} {}'.format(i, num_queries))
    ground_truth = set(brute_force_query(corpus[ix, :], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

    for r in range(1, max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix, :], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end - start)
        precision[r].append(len(set(result['id']) & ground_truth) / 10.0)
        average_distance[r].append(result['distance'][1:].mean())


plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(average_distance[i]) for i in range(1, 17)], linewidth=4,
         label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(precision[i]) for i in range(1, 17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 17), [np.mean(query_time[i]) for i in range(1, 17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

# effect of number of random vectors
precision = {i: [] for i in range(5, 20)}
average_distance = {i: [] for i in range(5, 20)}
query_time = {i: [] for i in range(5, 20)}
num_candidates_history = {i: [] for i in range(5, 20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix, :], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

for num_vector in range(5, 20):
    print('num_vector = {}'.format(num_vector))
    model = train_lsh(corpus, num_vector, seed=143)

    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix, :], model, k=10, max_search_radius=3)
        end = time.time()

        query_time[num_vector].append(end - start)
        precision[num_vector].append(len(set(result['id']) & ground_truth[ix]) / 10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)

plt.figure(figsize=(7, 4.5))
plt.plot(range(5, 20), [np.mean(average_distance[i]) for i in range(5, 20)], linewidth=4,
         label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in range(5, 20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in range(5, 20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7, 4.5))
plt.plot(range(5, 20), [np.mean(num_candidates_history[i]) for i in range(5, 20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()