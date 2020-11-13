import turicreate as tc
import numpy as np

sales = tc.SFrame.read_csv('kc_house_data.csv')
sales['floors'] = sales['floors'].astype(int)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()

    return features_matrix, output_array


def predict_output(feature_matrix, weights):
    return np.dot(feature_matrix, weights)


def normalize_features(features_matrix):
    norms = np.linalg.norm(features_matrix, axis=0)
    normalized_features = features_matrix / norms
    return normalized_features, norms


features = ['sqft_living', 'bedrooms']
output = 'price'
feature_matrix, output = get_numpy_data(sales, features, output)
feature_matrix, norms = normalize_features(feature_matrix)

# assign a random set of initial values of ro[i]
weights = np.array([1, 4, 1])
prediction = predict_output(feature_matrix, weights)
ro = [0 for j in range(feature_matrix.shape[1])]

for i in range(feature_matrix.shape[1]):
    ro[i] = sum(feature_matrix[:, i]*(output - prediction + weights[i]*feature_matrix[:, i]))


def in_l1range(ro, penalty):
    return -penalty/2 <= ro <= penalty/2


for l1_penalty in [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]:
    print('L1 Penalty: {} {}', in_l1range(ro[1], l1_penalty), in_l1range(ro[2], l1_penalty))


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)

    # compute ro[i]
    ro_i = sum(feature_matrix[:, i]*(output - prediction + weights[i]*feature_matrix[:, i]))

    # updates
    if i == 0:
        w = ro_i
    elif ro_i < -l1/2:
        w = ro_i + l1/2
    elif ro_i > l1/2:
        w = ro_i - l1/2
    else:
        w = 0

    return w


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    D = feature_matrix.shape[1]
    weights = np.array(initial_weights)
    change = np.array(initial_weights) * 0.0
    converged = False

    while not converged:
        for i in range(D):
            w = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            change[i] = np.abs(w - weights[i])
            weights[i] = w

        max_change = max(change)

        if max_change < tolerance:
            converged = True

    return weights


features = ['sqft_living', 'bedrooms']
output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

feature_matrix, output = get_numpy_data(sales, features, output)
normalized_feature_matrix, norms = normalize_features(feature_matrix)

# lasso coordinate descent
weights = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)

print(weights)

rss = sum((output - predict_output(normalized_feature_matrix, weights))**2)
print(rss)

# evaluate LASSO fit with more features
train_data, test_data = sales.random_split(.8, seed=0)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
output = 'price'

feature_matrix, output = get_numpy_data(train_data, features, output)
normalized_feature_matrix, norms = normalize_features(feature_matrix)

initial_weights = np.zeros(len(features) + 1)
l1_penalties = [1e7, 1e8, 1e4]
weights_list = []

for l1_penalty in l1_penalties:
    if l1_penalty == 1e4:
        tolerance = 5e5
    else:
        tolerance = 1.0

    w = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
    weights_list.append(w)

    print(l1_penalty)
    for (k, v) in dict(zip(['constant'] + features, w)).items():
        if v != 0.0:
            print('{} {}'.format(k, v))
    print()

# rescaling learned weights
for i in range(len(weights_list)):
    weights_list[i] = weights_list[i] / norms

# evaluate the model
test_feature_matrix, test_output = get_numpy_data(test_data, features, 'price')
for i in range(len(weights_list)):
    prediction = predict_output(test_feature_matrix, weights_list[i])
    rss = sum((test_output - prediction)**2)
    print('rss for {}: {}'.format(l1_penalties[i], rss))




