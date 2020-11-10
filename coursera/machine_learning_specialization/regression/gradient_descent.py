import turicreate as tc
import numpy as np
from math import sqrt


def get_numpy_data(data_sframe, features, output):

    data_sframe['constant'] = 1

    features = ['constant'] + features
    feature_sframe = data_sframe[features]
    features_matrix = feature_sframe.to_numpy()

    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()

    return features_matrix, output_array


def predict_outcome(feature_matrix, weights):
    return np.dot(feature_matrix, weights)


def feature_derivative(errors, feature):
    return 2*np.dot(feature, errors)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):

    converged = False
    weights = np.array(initial_weights)

    while not converged:

        predictions = predict_outcome(feature_matrix, weights)
        errors = predictions - output

        gradient_magnitude = 0

        for i in range(len(weights)):
            partial = feature_derivative(errors, feature_matrix[:, i])
            gradient_magnitude = gradient_magnitude + partial**2
            weights[i] = weights[i] - step_size*partial

        if sqrt(gradient_magnitude) < tolerance:
            converged = True

    return weights


train_data = tc.SFrame.read_csv('kc_house_train_data.csv')
test_data = tc.SFrame.read_csv('kc_house_test_data.csv')

feature_matrix, output = get_numpy_data(train_data, ['sqft_living'], 'price')
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(weights)

test_feature_matrix, test_output = get_numpy_data(test_data, ['sqft_living'], 'price')
predictions = predict_outcome(test_feature_matrix, weights)
print(predictions[0])
print(test_output[0])

rss = sum((test_output - predictions)**2)
print(rss)

feature_matrix, output = get_numpy_data(train_data, ['sqft_living', 'sqft_living15'], 'price')
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(weights)

test_feature_matrix, test_output = get_numpy_data(test_data, ['sqft_living', 'sqft_living15'], 'price')
predictions = predict_outcome(test_feature_matrix, weights)
print(predictions[0])
print(test_output[0])

rss = sum((test_output - predictions)**2)
print(rss)

