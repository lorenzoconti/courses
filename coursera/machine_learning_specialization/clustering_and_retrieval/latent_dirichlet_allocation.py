import turicreate as tc
import matplotlib.pyplot as plt
import numpy as np

# load the dataset
wiki = tc.SFrame('data/people_wiki.gl/')

import turicreate as tc

# load the dataset
wiki = tc.SFrame('people_wiki.sframe')

# create the bag of word features
wiki_docs = tc.text_analytics.count_words(wiki['text'])
# remove stopping words that don't help distinguish between documents
wiki_docs = wiki_docs.dict_trim_by_keys(tc.text_analytics.stop_words(), exclude=True)

# model fitting and interpretation
topic_model = tc.topic_model.create(wiki_docs, num_topics=10, num_iterations=200)

# show the hyper parameter settings (alpha, beta=gamma, K=number of topics)
print(topic_model)

# load a fitted topic model
topic_model = tc.load_model('topic_models/lda_assignment_topic_model')
print(topic_model.get_topics(topic_ids=[0]).sort('score', ascending=False))


# load a fitted topic model
topic_model = tc.load_model('data/topic_models/lda_assignment_topic_model')

# identifying topic themes by top words
print(topic_model.get_topics())

# what is the sum of the probabilities assigned to the top 50 words in the 3rd topic?
print(topic_model.get_topics([2], num_words=50))
print(topic_model.get_topics([2], num_words=50)['score'].sum())

print([x['words'] for x in topic_model.get_topics(output_type='topic_words', num_words=10)])

themes = ['science and research', 'team sports', 'music, TV, and film', 'American college and politics',
          'general politics', 'art and publishing', 'Business', 'international athletics',
          'Great Britain and Australia', 'international music']

# measuring the importance of top words
for i in range(10):
    plt.plot(range(100), topic_model.get_topics(topic_ids=[i], num_words=100)['score'])
plt.xlabel('Word rank')
plt.ylabel('Probability')
plt.title('Probabilities of Top 100 Words in each Topic')

# in the above plot, each line corresponds to one of our ten topics. Notice how for each topic, the weights drop off
# sharply as we move down the ranked list of most important words. This shows that the top 10-20 words in each topic
# are assigned a much greater weight than the remaining words - and remember from the summary of our topic model that
# our vocabulary has 547462 words in total!

# next we plot the total weight assigned by each topic to its top 10 words:
top_probs = [sum(topic_model.get_topics(topic_ids=[i], num_words=10)['score']) for i in range(10)]

ind = np.arange(10)
width = 0.5

fig, ax = plt.subplots()

ax.bar(ind - (width / 2), top_probs, width)
ax.set_xticks(ind)

plt.xlabel('Topic')
plt.ylabel('Probability')
plt.title('Total Probability of Top 10 Words in each Topic')
plt.xlim(-0.5, 9.5)
plt.ylim(0, 0.15)
plt.show()

# Here we see that, for our topic model, the top 10 words only account for a small fraction (in this case, between
# 5% and 13%) of their topic's total probability mass. So while we can use the top words to identify broad themes for
# each topic, we should keep in mind that in reality these topics are more complex than a simple 10-word summary.

# Finally, we observe that some 'junk' words appear highly rated in some topics despite our efforts to remove unhelpful
# words before fitting the model; for example, the word 'born' appears as a top 10 word in three different topics,
# but it doesn't help us describe these topics at all.

# topic distributions for some example documents

# LDA allows for mixed membership, which means that each document can partially belong to several different topics.
# For each document, topic membership is expressed as a vector of weights that sum to one; the magnitude of each weight
# indicates the degree to which the document represents that particular topic.

# topic distributions for documents can be obtained using turicreate's predict() function.
# turicreate uses a collapsed Gibbs sampler similar to the one described in the video lectures, where only the word
# assignments variables are sampled.
# To get a document-specific topic proportion vector post-facto, predict() draws this vector from the conditional
# distribution given the sampled word assignments in the document.
# Notice that, since these are draws from a distribution over topics that the model has learned, we will get slightly
# different predictions each time we call this function on a document - we can see this below, where we predict the
# topic distribution for the article on Barack Obama:

# In[21]:

obama = tc.SArray([wiki_docs[int(np.where(wiki['name'] == 'Barack Obama')[0])]])
pred1 = topic_model.predict(obama, output_type='probability')
pred2 = topic_model.predict(obama, output_type='probability')
print(tc.SFrame({'topics': themes, 'predictions (first draw)': pred1[0],
                 'predictions (second draw)': pred2[0]}))


# to get a more robust estimate of the topics for each document, we can average a large number of predictions for the
# same document:
def average_predictions(model, test_document, num_trials=100):
    avg_preds = np.zeros(model.num_topics)
    for i in range(num_trials):
        avg_preds += model.predict(test_document, output_type='probability')[0]
    avg_preds = avg_preds / num_trials
    result = tc.SFrame({'topics': themes, 'average predictions': avg_preds})
    result = result.sort('average predictions', ascending=False)
    return result


print(average_predictions(topic_model, obama, 100))

# what is the topic most closely associated with the article about former US President George W. Bush?
george_w_bush = tc.SArray([wiki_docs[int(np.where(wiki['name'] == 'George W. Bush')[0])]])
print(average_predictions(topic_model, george_w_bush, 100))

# what are the top 3 topics corresponding to the article about English football (soccer) player Steven Gerrard?
gerrard = tc.SArray([wiki_docs[int(np.where(wiki['name'] == 'Steven Gerrard')[0])]])
print(average_predictions(topic_model, gerrard, 100))

# comparing LDA to nearest neighbors for document retrieval

# create the LDA topic distribution representation for each document
wiki['lda'] = topic_model.predict(wiki_docs, output_type='probability')

# add the TF-IDF document representations
wiki['word_count'] = tc.text_analytics.count_words(wiki['text'])
wiki['tf_idf'] = tc.text_analytics.tf_idf(wiki['word_count'])

# for each of our two different document representations, compute a brute-force nearest neighbors model
model_tf_idf = tc.nearest_neighbors.create(wiki, label='name', features=['tf_idf'], mehtod='brute_force',
                                           distance='cosine')
model_lda_rep = tc.nearest_neighbors.create(wiki, label='name', features=['lda'], method='brute_force',
                                            distance='cosine')

print(model_tf_idf.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10))

print(model_lda_rep.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10))

# notice that that there is no overlap between the two sets of top 10 nearest neighbors.
# This doesn't necessarily mean that one representation is better or worse than the other, but rather that they are
# picking out different features of the documents.

# with TF-IDF, documents are distinguished by the frequency of uncommon words.
# Since similarity is defined based on the specific words used in the document, documents that are "close" under TF-IDF
# tend to be similar in terms of specific details.

# the LDA representation, on the other hand, defines similarity between documents in terms of their topic distributions.
# This means that documents can be "close" if they share similar themes, even though they may not share many of the same
# keywords. For the article on Paul Krugman, we expect the most important topics to be 'American college and politics'
# and 'science and research'.
# As a result, we see that the top 10 nearest neighbors are academics from a wide variety of fields, including
# literature, anthropology, and religious studies.

# compute the 5000 nearest neighbors for American baseball player Alex Rodriguez.
# for what value of k is Mariano Rivera the k-th nearest neighbor to Alex Rodriguez?

# compute the 5000 nearest neighbors for American baseball player Alex Rodriguez. For what value of k is Mariano Rivera
# the k-th nearest neighbor to Alex Rodriguez?
rodriguez_tfidf_neighbors = model_tf_idf.query(wiki[wiki['name'] == 'Alex Rodriguez'],
                                               label='name', k=5000)
print("type(rodriguez_tfidf_neighbors): %s" % (type(rodriguez_tfidf_neighbors)))
print(rodriguez_tfidf_neighbors[:5])

print(rodriguez_tfidf_neighbors[rodriguez_tfidf_neighbors['reference_label'] == "Mariano Rivera"])

# compute the 5000 nearest neighbors for American baseball player Alex Rodriguez.
# for what value of k is Mariano Rivera the k-th nearest neighbor to Alex Rodriguez?
rodriguez_lda_neighbors = model_lda_rep.query(wiki[wiki['name'] == 'Alex Rodriguez'], label='name', k=5000)
print("type(rodriguez_lda_neighbors): %s" % (type(rodriguez_lda_neighbors)))
print(rodriguez_lda_neighbors[:5])
print(rodriguez_lda_neighbors[rodriguez_lda_neighbors['reference_label'] == "Mariano Rivera"])

# understanding the role of LDA model hyperparameters
# alpha is a parameter of the prior distribution over topic weights in each document,
# while gamma is a parameter of the prior distribution over word weights in each topic.

# alpha and gamma can be thought of as smoothing parameters when we compute how much each document "likes" a topic
# (in the case of alpha) or how much each topic "likes" a word (in the case of gamma).
# In both cases, these parameters serve to reduce the differences across topics or words in terms of these calculated
# preferences; alpha makes the document preferences "smoother" over topics, and gamma makes the topic preferences
# "smoother" over words.

tpm_low_alpha = tc.load_model('data/topic_models/lda_low_alpha')
tpm_high_alpha = tc.load_model('data/topic_models/lda_high_alpha')

# changing the hyperparameter alpha

# since alpha is responsible for smoothing document preferences over topics, the impact of changing its value should be
# visible when we plot the distribution of topic weights for the same document under models fit with different alpha
# values.
a = np.sort(tpm_low_alpha.predict(obama, output_type='probability')[0])[::-1]
b = np.sort(topic_model.predict(obama, output_type='probability')[0])[::-1]
c = np.sort(tpm_high_alpha.predict(obama, output_type='probability')[0])[::-1]
ind = np.arange(len(a))
width = 0.3


def param_bar_plot(a, b, c, ind, width, ylim, param, xlab, ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    b1 = ax.bar(ind, a, width, color='lightskyblue')
    b2 = ax.bar(ind + width, b, width, color='lightcoral')
    b3 = ax.bar(ind + (2 * width), c, width, color='gold')

    ax.set_xticks(ind + width)
    ax.set_xticklabels(range(10))
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(0, ylim)
    ax.legend(handles=[b1, b2, b3], labels=['low ' + param, 'original model', 'high ' + param])

    plt.tight_layout()
    plt.show()


param_bar_plot(a, b, c, ind, width, ylim=1.0, param='alpha',
               xlab='Topics (sorted by weight of top 100 words)', ylab='Topic Probability for Obama Article')

krugman = tc.SArray([wiki_docs[int(np.where(wiki['name'] == 'Paul Krugman')[0])]])
print(average_predictions(tpm_low_alpha, krugman, 100))

# how many topics are assigned a weight greater than 0.3 or less than 0.05 for the article on Paul Krugman in the
# high alpha model?

print(average_predictions(tpm_high_alpha, krugman, 100))

# changing the hyperparameter gamma
# we expect to be able to visualize the impact of changing gamma by plotting word weights for each topic.
# In this case, however, there are far too many words in our vocabulary to do this effectively.
# Instead, we'll plot the total weight of the top 100 words and bottom 1000 words for each topic.
# Below, we plot the (sorted) total weights of the top 100 words and bottom 1000 from each topic in the high, original,
# and low gamma models.

tpm_low_gamma = tc.load_model('data/topic_models/lda_low_gamma')
tpm_high_gamma = tc.load_model('data/topic_models/lda_high_gamma')

a_top = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
b_top = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
c_top = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]

a_bot = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i],
                                              num_words=547462)[-1000:]['score']) \
                 for i in range(10)])[::-1]
b_bot = np.sort([sum(topic_model.get_topics(topic_ids=[i],
                                            num_words=547462)[-1000:]['score']) \
                 for i in range(10)])[::-1]
c_bot = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i],
                                               num_words=547462)[-1000:]['score']) \
                 for i in range(10)])[::-1]

ind = np.arange(len(a))
width = 0.3

param_bar_plot(a_top, b_top, c_top, ind, width, ylim=0.6, param='gamma',
               xlab='Topics (sorted by weight of top 100 words)',
               ylab='Total Probability of Top 100 Words')

param_bar_plot(a_bot, b_bot, c_bot, ind, width, ylim=0.0002, param='gamma',
               xlab='Topics (sorted by weight of bottom 1000 words)',
               ylab='Total Probability of Bottom 1000 Words')

# we can see that the low gamma model results in higher weight placed on the top words and lower weight placed on the
# bottom words for each topic, while the high gamma model places relatively less weight on the top words and more weight
# on the bottom words. Thus increasing gamma results in topics that have a smoother distribution of weight across all
# the words in the vocabulary.

# compute the number of words required to make a list with total probability 0.5.
# What is the average number of words required across all topics?
print(tpm_low_gamma.get_topics(cdf_cutoff=0.5).print_rows(num_rows=50))

# for each topic of the high gamma model, compute the number of words required to make a list with total probability 0.5
print(tpm_high_gamma.get_topics(cdf_cutoff=0.5).print_rows(num_rows=50))




