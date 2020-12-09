import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

wiki = pd.read_csv('people_wiki.csv')
print(wiki.head())


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)


word_count = load_sparse_csr('people_wiki_word_count.npz')
map_index_to_word = pd.read_json('people_wiki_map_index_to_word.json', typ='series')


def count_words(X, voc):
    """
    X: the return matrix of CountVectorizer.transform
    voc : vect.vocabulary_
    """
    rvoc = dict((v, k) for k, v in voc.iteritems())
    print(X.shape[0])

    def count(row_id):
        dic = dict()
        print(row_id)
        # extract the indices (columns) of a sparse matrix given a specific row
        for ind in X[row_id, :].indices:
            dic[rvoc[ind]] = X[row_id, ind]
        return dic

    word_count = list(map(count, range(0, X.shape[0])))
    return word_count


word_counts = count_words(word_count, map_index_to_word)
wiki['word_count'] = word_counts

print(wiki.head())

# find nearest neighbors
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

print(wiki[wiki['name'] == 'Barack Obama'])

distances, indices = model.kneighbors(word_count[35817, :], n_neighbors=10)

neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
neighbors.set_index('id', inplace=True)
print(wiki.join(neighbors, how='right').nsmallest(10, 'distance')[['name', 'distance']])


# let's find out why Francisco Barrio was considered a close neighbor of Obama.
# To do this, let's look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages.
def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = pd.DataFrame({'word': row['word_count'].tolist()[0].keys(),
                                     'count': row['word_count'].tolist()[0].values()}, )
    word_count_table.set_index('word', inplace=True)
    return word_count_table.sort_values('count', ascending=False)


obama_words = top_words('Barack Obama')
print(obama_words.head())

barrio_words = top_words('Francisco Barrio')
print(barrio_words.head())

combined_words = obama_words.join(barrio_words, how='inner', lsuffix='_obama', rsuffix='_barrio')
print(combined_words.head())

combined_words = combined_words.rename(columns={'count_obama': 'Obama', 'count_barrio': 'Barrio'})
print(combined_words.head())

print(combined_words.sort_values('Obama', ascending=False).head())

# among the words that appear in both Barack Obama and Francisco Barrio, take the 5 that appear most frequently in Obama
# how many of the articles in the Wikipedia dataset contain all of those 5 words?
common_words = set(combined_words.sort_values('Obama', ascending=False)[0:5].index)


def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)  # YOUR CODE HERE


wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print(sum(wiki['has_top_words']))

print(euclidean_distances(word_count[wiki[wiki['name'] == 'Barack Obama'].index.values[0], :],
                          word_count[wiki[wiki['name'] == 'George W. Bush'].index.values[0], :]))
print(euclidean_distances(word_count[wiki[wiki['name'] == 'Barack Obama'].index.values[0], :],
                          word_count[wiki[wiki['name'] == 'Joe Biden'].index.values[0], :]))
print(euclidean_distances(word_count[wiki[wiki['name'] == 'Joe Biden'].index.values[0], :],
                          word_count[wiki[wiki['name'] == 'George W. Bush'].index.values[0], :]))

bush_words = top_words('George W. Bush')
combined_words = obama_words.join(bush_words, how='inner', rsuffix='.1')
print(combined_words.sort_values('count', ascending=False)[0:10])

# tf-idf to the rescue
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
tf_idfs = count_words(tf_idf, map_index_to_word)
wiki['tf_idf'] = tf_idfs

model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)

distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)

neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
neighbors.set_index('id', inplace=True)
print(wiki.join(neighbors, how='right').nsmallest(10, 'distance')[['name', 'distance']])


def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = pd.DataFrame({'word': row['tf_idf'].tolist()[0].keys(),
                                     'weight': row['tf_idf'].tolist()[0].values()}, )
    return word_count_table.sort_values('weight', ascending=False)


obama_tf_idf = top_words_tf_idf('Barack Obama')
obama_tf_idf.head()

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
schiliro_tf_idf.head()

obama_tf_idf.set_index('word', inplace=True)
schiliro_tf_idf.set_index('word', inplace=True)
combined_words = obama_tf_idf.join(schiliro_tf_idf, how='inner', rsuffix='.1')
combined_words.sort_values('weight', ascending=False, inplace=True)
combined_words.head(10)

common_words = set(combined_words.iloc[0:5].index)


def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)


wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print(sum(wiki['has_top_words']))

print(euclidean_distances(tf_idf[wiki[wiki['name'] == 'Barack Obama'].index],
                          tf_idf[wiki[wiki['name'] == 'Joe Biden'].index]))


def compute_length(text):
    return len(text.split(' '))


wiki['length'] = wiki['text'].apply(compute_length)

# compute 100 nearest neighbors and display their lengths
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
neighbors.set_index('id', inplace=True)
nearest_neighbors_euclidean = wiki.join(neighbors, how='inner')[['name', 'length', 'distance']].sort_values('distance')
nearest_neighbors_euclidean.head(10)

# to see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram of
# the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents
plt.figure(figsize=(10.5, 4.5))

plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', density=True, label='Entire Wikipedia',
         zorder=3, alpha=0.8)

plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', density=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)

plt.axvline(x=wiki[wiki['name'] == 'Barack Obama']['length'].values[0], color='k', linestyle='--', linewidth=4,
            label='Length of Barack Obama', zorder=2)

plt.axvline(x=wiki[wiki['name'] == 'Joe Biden']['length'].values[0], color='g', linestyle='--', linewidth=4,
            label='Length of Joe Biden', zorder=1)

plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size': 15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

# Relative to the rest of Wikipedia, nearest neighbors of Obama are overwhemingly short, most of them being shorter than
# 300 words. The bias towards short articles is not appropriate in this application as there is really no reason to
# favor short articles over long articles (they are all Wikipedia articles, after all). Many of the Wikipedia articles
# are 300 words or more, and both Obama and Biden are over 300 words long.

# To remove this bias, we turn to cosine distances:
# Cosine distances let us compare word distributions of two articles of varying lengths.


model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tf_idf)

distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
neighbors.set_index('id', inplace=True)
nearest_neighbors_cosine = wiki.join(neighbors, how='inner')[['name', 'length', 'distance']].sort_values('distance')
print(nearest_neighbors_cosine.head())

plt.figure(figsize=(10.5, 4.5))

plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', density=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)

plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', density=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)

plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', density=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)

plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'].values[0], color='k', linestyle='--', linewidth=4,
            label='Length of Barack Obama', zorder=2)

plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'].values[0], color='g', linestyle='--', linewidth=4,
            label='Length of Joe Biden', zorder=1)

plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size': 15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.show()

# Cosine distances ignore all document lengths, which may be great in certain situations but not in others.
# For instance, consider the following (admittedly contrived) example.
# +--------------------------------------------------------+
# |                                             +--------+ |
# |  One that shall not be named                | Follow | |
# |  @username                                  +--------+ |
# |                                                        |
# |  Democratic governments control law in response to     |
# |  popular act.                                          |
# |                                                        |
# |  8:05 AM - 16 May 2016                                 |
# |                                                        |
# |  Reply   Retweet (1,332)   Like (300)                  |
# |                                                        |
# +--------------------------------------------------------+

# How similar is this tweet to Barack Obama's Wikipedia article?

tweet = {'act': 3.4597778278724887,
         'control': 3.721765211295327,
         'democratic': 3.1026721743330414,
         'governments': 4.167571323949673,
         'in': 0.0009654063501214492,
         'law': 2.4538226269605703,
         'popular': 2.764478952022998,
         'response': 4.261461747058352,
         'to': 0.04694493768179923}

word_indices = [map_index_to_word[map_index_to_word.index == word][0] for word in tweet.keys()]
tweet_tf_idf = csr_matrix((list(tweet.values()), ([0]*len(word_indices), word_indices)), shape=(1, tf_idf.shape[1]))

obama_tf_idf = tf_idf[35817]
print(cosine_distances(obama_tf_idf, tweet_tf_idf))

distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})
neighbors.set_index('id', inplace=True)
nearest_neighbors_cosine = wiki.join(neighbors, how='inner')[['name', 'distance']].sort_values('distance')
print(nearest_neighbors_cosine.head(10))

# With cosine distances, the tweet is "nearer" to Barack Obama than everyone else, except for Joe Biden! This probably
# is not something we want. If someone is reading the Barack Obama Wikipedia page, would you want to recommend they read
# this tweet? Ignoring article lengths completely resulted in nonsensical results. In practice, it is common to enforce
# maximum or minimum document lengths. After all, when someone is reading a long article from The Atlantic, you wouldn't
# recommend him/her a tweet.
