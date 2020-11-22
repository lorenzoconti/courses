import turicreate as tc
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

products = tc.SFrame.read_csv('amazon_baby_subset.csv')

with open('important_words.json') as file:
    important_words = json.load(file)

products = products.fillna('review', '')


# remove punctuation
def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    features_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return features_matrix, label_array


train_data, validation_data = products.random_split(.8, seed=2)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_validation, sentiment_validation = get_numpy_data(validation_data, important_words, 'sentiment')


def predict_probabilty(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1/(1+np.exp(-score))
    return predictions


def feature_derivative_with_l2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    derivative = np.dot(np.transpose(errors), feature)
    if not feature_is_constant:
        derivative -= 2 * l2_penalty * coefficient
    return derivative


def compute_log_likelihood_with_l2(feature_matrix, sentiment, coefficients, l2):
    indicator = sentiment == 1
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum(
        (np.transpose(np.array([indicator]))-1)*scores - np.log(1 + np.exp(-scores))) - l2 * np.sum(coefficients[1:]**2)
    return lp


def logistic_regression_with_l2(feature_matrix, sentiment, initial_coefficients, l2, step_size, max_iter):

    coefficients = np.array(initial_coefficients)
    lplist = []

    for itr in range(max_iter):

        predictions = predict_probabilty(feature_matrix, coefficients)
        indicator = sentiment == 1
        errors = np.transpose(np.array([indicator])) - predictions

        for j in range(len(coefficients)):

            derivative = feature_derivative_with_l2(errors, feature_matrix[:, j], coefficients[j], l2, j == 0)
            coefficients[j] += step_size*derivative

        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:

            lp = compute_log_likelihood_with_l2(feature_matrix, sentiment, coefficients, l2)
            lplist.append(lp)
            print('iteration {}: log likelihood of observed labels = {}'.format(itr, lp))

    x = [i for i in range(len(lplist))]
    plt.figure()
    plt.plot(x, lplist, 'ro')
    plt.title(l2)
    plt.show()

    return coefficients


l2_penalties = [0, 4, 10, 1e2, 1e3, 1e5]
model_coefficients = []
initial_coefficients = np.zeros((194, 1))
step_size = 5e-6
max_iter = 501

for l2 in l2_penalties:

    model_coefficients.append(logistic_regression_with_l2(feature_matrix_train, sentiment_train, initial_coefficients,
                                                          l2, step_size, max_iter))

word_coefficient_tuples = [(word, coefficient)
                           for word, coefficient in zip(important_words, list(model_coefficients[0][1:]))]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x: x[1], reverse=True)

positive_words = []
for word in word_coefficient_tuples[:5]:
    positive_words.append(word[0])

negative_words = []
for word in word_coefficient_tuples[-5:]:
    negative_words.append(word[0])

table = pd.DataFrame(data=[coeff.flatten() for coeff in model_coefficients], index=l2_penalties,
                     columns=['intercept']+important_words)

plt.rcParams['figure.figsize'] = 10, 6


def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')

    xx = l2_penalty_list
    plt.plot(xx, [0.] * len(xx), '--', lw=1, color='k')

    table_positive_words = table[positive_words]
    table_negative_words = table[negative_words]

    for i, value in enumerate(positive_words):
        color = cmap_positive(0.8 * ((i + 1) / (len(positive_words) * 1.2) + 0.15))
        plt.plot(xx, table_positive_words[value].to_numpy().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)

    for i, value in enumerate(negative_words):
        color = cmap_negative(0.8 * ((i + 1) / (len(negative_words) * 1.2) + 0.15))
        plt.plot(xx, table_negative_words[value].to_numpy().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)

    plt.legend(loc='best', ncol=3, prop={'size': 16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty lambda')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()


make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=l2_penalties)

training_accuracy = []
for c in model_coefficients:
    predictions = predict_probabilty(feature_matrix_train, c)
    corrects = np.sum([(np.transpose(predictions.flatten()) > 0.5)[i] == (np.array(sentiment_train) > 0)[i]
                       for i in range(len(predictions))])
    training_accuracy.append(corrects / len(sentiment_train))

plt.figure()
plt.plot([x for x in range(len(model_coefficients))], training_accuracy, 'ro')
plt.title('training accuracy')
plt.xlabel('L2 penalty')
plt.ylabel('training_accuracy')
plt.ylim((0, 1))
plt.show()

validation_accuracy = []
for c in model_coefficients:
    predictions = predict_probabilty(feature_matrix_validation, c)
    corrects = np.sum([(np.transpose(predictions.flatten()) > 0.5)[i] == (np.array(sentiment_validation) > 0)[i]
                       for i in range(len(predictions))])
    validation_accuracy.append(corrects / len(sentiment_validation))

plt.figure()
plt.plot([x for x in range(len(model_coefficients))], validation_accuracy, 'ro')
plt.title('validation accuracy')
plt.xlabel('L2 penalty')
plt.ylabel('validation_accuracy')
plt.ylim((0, 1))
plt.show()

plt.figure()
plt.plot([x for x in range(len(model_coefficients))], training_accuracy, 'g')
plt.plot([x for x in range(len(model_coefficients))], validation_accuracy, 'r')