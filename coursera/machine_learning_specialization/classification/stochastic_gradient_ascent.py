import turicreate as tc
import json
import numpy as np
import matplotlib.pyplot as plt

products = tc.SFrame.read_csv('amazon_baby_subset.csv')

with open('important_words.json') as f:
    important_words = json.load(f)

important_words = [str(s) for s in important_words]
products = products.fillna('review', '')


def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

train_data, validation_data = products.random_split(.8, seed=1)


def get_numpy_data(dataframe, features, label):
    dataframe['intercept'] = 1
    features = ['intercept'] + features
    features_frame = dataframe[features]
    features_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return features_matrix, label_array


feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_validation, sentiment_validation = get_numpy_data(validation_data, important_words, 'sentiment')


def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1./(1.+np.exp(-score))
    return predictions


def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature)
    return derivative


def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = sentiment == 1
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))

    # prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]

    lp = np.sum((indicator-1)*scores - logexp) / len(feature_matrix)

    # added a 1/N term which averages the log likelihood across all the data points and makes it esier to compare
    # stochastic gradient ascent with batch gradient ascent

    # compute_log_likelihood in 'logistic_regression_scratcg.py'
    # lp = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1 + np.exp(-scores)))
    return lp


def logistic_regression_sg(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):

    log_likelihood_all = []
    coefficients = np.array(initial_coefficients)

    np.random.seed(seed=1)

    # shuffle data before starting
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation, :]
    sentiment = sentiment[permutation]

    i = 0

    for itr in range(max_iter):

        # predict P(y_i = +1 | x_i, w)
        predictions = predict_probability(feature_matrix[i:i+batch_size, :], coefficients)

        indicator = sentiment[i:i+batch_size] == 1
        errors = indicator - predictions

        for j in range(len(coefficients)):

            derivative = feature_derivative(errors, feature_matrix[i:i+batch_size, j])

            coefficients[j] += 1./batch_size*step_size*derivative

        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size, :], sentiment[i:i+batch_size], coefficients)
        log_likelihood_all.append(lp)

        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0 \
                or itr == max_iter-1:
            data_size = len(feature_matrix)
            print('Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' %
                  (int(np.ceil(np.log10(max_iter))), itr, int(np.ceil(np.log10(data_size))), i,
                   int(np.ceil(np.log10(data_size))), i+batch_size, lp))

        # if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation, :]
            sentiment = sentiment[permutation]
            i = 0

    return coefficients, log_likelihood_all


initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = 1
max_iter = 10

coefficients, log_likelihood = logistic_regression_sg(feature_matrix_train, sentiment_train, initial_coefficients,
                                                      step_size, batch_size, max_iter)

plt.plot(range(0, 10), log_likelihood)

initial_coefficients = np.zeros(194)
step_size = 1e-1
batch_size = 100
num_passes = 10
max_iter = num_passes * int(len(feature_matrix_train)/batch_size)

coefficients, log_likelihood = logistic_regression_sg(feature_matrix_train, sentiment_train, initial_coefficients,
                                                      step_size, batch_size, max_iter)

plt.plot(range(0, max_iter), log_likelihood)


def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9, 5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all),
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size': 14})


plt.figure()
make_plot(log_likelihood, len_data=len(feature_matrix_train), batch_size=batch_size,
          label='stochastic gradient, step_size=1e-1')

plt.figure()
make_plot(log_likelihood, len_data=len(feature_matrix_train), batch_size=batch_size, smoothing_window=30,
          label='stochastic gradient, step_size=1e-1')

# stochastic gradient ascent vs batch gradient ascent
step_size = 1e-1
batch_size = 100
num_passes = 200
max_iter = num_passes * int(len(feature_matrix_train) / batch_size)

_, log_likelihood_sgd = logistic_regression_sg(feature_matrix_train, sentiment_train,
                                                              initial_coefficients, step_size, batch_size, max_iter)


step_size = 5e-1
batch_size = len(feature_matrix_train)
num_passes = 200
max_iter = num_passes * int(len(feature_matrix_train) / batch_size)

_, log_likelihood_bgd = logistic_regression_sg(feature_matrix_train, sentiment_train,
                                                              initial_coefficients, step_size, batch_size, max_iter)

plt.figure()
make_plot(log_likelihood_sgd, len_data=len(feature_matrix_train), batch_size=100, smoothing_window=30,
          label='stochastic gradient, step_size=1e-1')
make_plot(log_likelihood_bgd, len_data=len(feature_matrix_train), batch_size=batch_size, smoothing_window=1,
          label='stochastic gradient, step_size=1e-1')

batch_size = 100
num_passes = 10
max_iter = num_passes * int(len(feature_matrix_train) / batch_size)

step_sizes = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

coefficients_sgd = {}
log_likelihood_sgd = {}

plt.figure()

for step_size in step_sizes:

    coefficients_sgd[step_size], log_likelihood_sgd[step_size] = \
        logistic_regression_sg(feature_matrix_train, sentiment_train, initial_coefficients, step_size, batch_size,
                               max_iter)
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=batch_size, smoothing_window=30,
              label='step_size={}'.format(step_size))

# remove step_size 1e2
plt.figure()
for step_size in np.logspace(-4, 2, num=7)[0:6]:
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=batch_size, smoothing_window=30,
              label='step_size={}'.format(step_size))

