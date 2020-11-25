import turicreate as tc
import numpy as np
import json

products = tc.SFrame.read_csv('amazon_baby_subset.csv')

print((products['sentiment'] == 1).sum())
print((products['sentiment'] == -1).sum())
print(len(products['sentiment']))

with open('important_words.json') as file:
    important_words = json.load(file)

print(important_words[:3])

products = products.fillna('review', '')


# remove punctuation
def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

# number of product reviews that contain the word perfect
print(sum(products['perfect'] >= 1))


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    features_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return features_matrix, label_array


feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')

print(feature_matrix.shape)


def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1/(1+np.exp(-score))
    return predictions


def feature_derivative(errors, feature):
    derivative = np.dot(np.transpose(errors), feature)
    return derivative


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = sentiment == 1
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1 + np.exp(-scores)))
    return lp


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):

    coefficients = np.array(initial_coefficients)
    lplist = []

    for itr in range(max_iter):

        predictions = predict_probability(feature_matrix, coefficients)
        indicator = sentiment == 1
        errors = np.transpose(np.array([indicator])) - predictions

        for j in range(len(coefficients)):

            derivative = feature_derivative(errors, feature_matrix[:,j])
            coefficients[j] += step_size*derivative

        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lplist.append(compute_log_likelihood(feature_matrix, sentiment, coefficients))
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration {}: log likelihood of observed labels = {} {}'.format(int(np.ceil(np.log10(max_iter))),
                                                                                   itr, lp))

    import matplotlib.pyplot as plt
    x = [i for i in range(len(lplist))]
    plt.plot(x, lplist, 'ro')
    plt.show()

    return coefficients


initial_coefficients = np.zeros((194, 1))
step_size = 1e-7
max_iter = 301

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter)

predictions = predict_probability(feature_matrix, coefficients)
num_positive = sum(predictions > 0.5)

score = np.dot(feature_matrix, coefficients)
print(sum(score > 0))

corrects = np.sum((np.transpose(predictions.flatten()) > 0.5) == np.array(products['sentiment'] > 0))

print('accuracy: {}'.format(corrects/len(products['sentiment'])))

coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]

word_coefficient_tuples = sorted(word_coefficient_tuples, key= lambda x: x[1], reverse=True)

print(word_coefficient_tuples[:10])
print(word_coefficient_tuples[len(word_coefficient_tuples)-11:len(word_coefficient_tuples)+1])





