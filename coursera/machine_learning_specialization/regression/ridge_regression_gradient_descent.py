import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv('kc_house_data.csv')


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe['price']
    output_array = output_sarray.to_numpy()
    return features_matrix, output_array


def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature)
    else:
        derivative = 2 * (np.dot(errors, feature) + l2_penalty * weight)
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2, max_iterations=100):

    weights = np.array(initial_weights)
    iterations = 0
    while iterations < max_iterations:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        for i in range(len(weights)):
            ft = feature_matrix[:, i]
            gradient = feature_derivative_ridge(errors, ft, weights[i], l2, i == 0)
            weights[i] = weights[i] - step_size*gradient
        iterations = iterations + 1
    return weights


features = ['sqft_living']
target = 'price'

train_data, test_data = sales.random_split(.8, seed=0)
feature_matrix, output = get_numpy_data(train_data, features, target)
feature_matrix_test, output_test = get_numpy_data(test_data, features, target)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations = 1000

weights = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0, max_iterations)
weights_l2 = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)

print('Unregularized weights: {}, Regolarized weigths: {}'.format(weights, weights_l2))

plt.plot(feature_matrix, output, 'k.',
         feature_matrix, predict_output(feature_matrix, weights), 'b-',
         feature_matrix, predict_output(feature_matrix, weights_l2), 'r-')

# rss on predictions with initial weights
residuals = output_test - predict_output(feature_matrix_test, initial_weights)
rss = sum(residuals**2)
print(rss)

# rss on predictions without l2 penalty
residuals = output_test - predict_output(feature_matrix_test, weights)
rss = sum(residuals**2)
print(rss)

# rss on preditcions with l2 penalty
residuals = output_test - predict_output(feature_matrix_test, weights_l2)
rss = sum(residuals**2)
print(rss)

# multiple regression
features = ['sqft_living', 'sqft_living15']
target = 'price'
feature_matrix, output = get_numpy_data(train_data, features, target)
feature_matrix_test, output_test = get_numpy_data(test_data, features, target)

initial_weights = np.array([0, 0, 0])
step_size = 1e-12
max_iterations = 1000

multiple_weights = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0,
                                                     max_iterations)

multiple_weights_l2 = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11,
                                                        max_iterations)

print('Unregularized weights: {}, Regolarized weigths: {}'.format(multiple_weights, multiple_weights_l2))

for w in [initial_weights, multiple_weights, multiple_weights_l2]:
    residuals = output_test - predict_output(feature_matrix_test, w)
    rss = sum(residuals ** 2)
    print(rss)

print(output_test[0] - predict_output(feature_matrix_test[0, :], multiple_weights))
print(output_test[0] - predict_output(feature_matrix_test[0, :], multiple_weights_l2))


