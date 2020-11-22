import turicreate as tc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

products = tc.SFrame.read_csv('amazon_baby.csv')
products = products.fillna('review', '')
# using pandas: products = products.fillna({'review': ''})


# remove punctuation
def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


products['review_clean'] = products['review'].apply(remove_punctuation)

# ignore reviews with rating=3 since they tend to have a neutral sentiment
products = products[products['rating'] != 3]

products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# learn vocabulary from the training data and assign columns to words
# then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# convert test data into a sparse matrix using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

# how many weights are greater then zero?
print(np.sum(sentiment_model.coef_ >= 0))

sample_test_data = test_data[10:13]
print(sample_test_data[0]['review'])
print(sample_test_data[1]['review'])

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)

print(scores)
print(sentiment_model.predict(sample_test_matrix))

print([1./(1+np.exp(-x)) for x in scores])

print(sentiment_model.classes_)
print(sentiment_model.predict_proba(sample_test_matrix))

# find the most positive and negative review
test_scores = sentiment_model.decision_function(test_matrix)

positive_idx = np.argsort(-test_scores)[:20]
print(test_scores[positive_idx[0]])
print(test_data[positive_idx[0]])

negative_idx = np.argsort(test_scores)[:20]
print(test_scores[negative_idx[0]])
print(test_data[negative_idx[0]])

# compute the accuracy
predictions = sentiment_model.predict(test_matrix)
corrects = np.sum(predictions == test_data['sentiment'])
total = len(test_data['sentiment'])

accuracy = corrects/total
print(accuracy)

# learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 'well', 'able', 'car', 'broke',
                     'less', 'even', 'waste', 'disappointed', 'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])

coefficients = pd.DataFrame({'word': significant_words, 'coefficient': simple_model.coef_.flatten()})
coefficients.sort_values(['coefficient'], ascending=False)

print(np.sum(simple_model.coef_ >= 0))

coefficients_sentiment_model = pd.DataFrame({'word': vectorizer.get_feature_names(),
                                             'coefficient': sentiment_model.coef_.flatten()})

intersection = pd.merge(coefficients, coefficients_sentiment_model, how='inner', on=['word'])
changed = intersection.query('coefficient_x > 0 & coefficient_y >0')
print(sum(coefficients['coefficient'] > 0) - len(changed))

# comparing models
# accuracy on training data
predictions = sentiment_model.predict(train_matrix)
corrects = np.sum(predictions == train_data['sentiment'])
total = len(train_data['sentiment'])

accuracy = corrects/total
print('accuracy on training data with complex model: {}'.format(accuracy))

predictions = simple_model.predict(train_matrix_word_subset)
corrects = np.sum(predictions == train_data['sentiment'])
total = len(train_data['sentiment'])

accuracy = corrects/total
print('accuracy on training data with a simpler model: {}'.format(accuracy))

# accuracy on test data
predictions = sentiment_model.predict(test_matrix)
corrects = np.sum(predictions == test_data['sentiment'])
total = len(test_data['sentiment'])

accuracy = corrects/total
print('accuracy on testing data with complex model: {}'.format(accuracy))

predictions = simple_model.predict(test_matrix_word_subset)
corrects = np.sum(predictions == test_data['sentiment'])
total = len(test_data['sentiment'])

accuracy = corrects/total
print('accuracy on testing data with a simpler model: {}'.format(accuracy))

# majority class prediction
positive_label = sum(test_data['sentiment'] > 0)
negative_label = sum(test_data['sentiment'] < 0)

baseline_accuracy = positive_label/(positive_label+negative_label)
print('baseline accuracy: ' + str(baseline_accuracy))

