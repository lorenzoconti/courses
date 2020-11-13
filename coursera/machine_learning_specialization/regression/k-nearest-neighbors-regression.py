import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv('kc_house_data_small.csv')


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()

    return features_matrix, output_array


def normalize_features(features_matrix):
    norms = np.linalg.norm(features_matrix, axis=0)
    normalized_features = features_matrix / norms
    return normalized_features, norms


train_val_data, test_data = sales.random_split(.8, seed=1)
train_data, validation_data = train_val_data.random_split(.8, seed=1)

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

features_train, output_train = get_numpy_data(train_data, features, 'price')
features_test, output_test = get_numpy_data(test_data, features, 'price')
features_validation, output_validation = get_numpy_data(validation_data, features, 'price')

features_train, norms = normalize_features(features_train)
features_test = features_test / norms
features_validation = features_validation / norms


def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))


dist_dict = {}

for i in range(10):
    d = dist(features_test[0], features_train[i])
    dist_dict[i] = d
    print('distance: ' + str(d))

sorted_dist_dict = {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}
print('min distance: ' + str(list(sorted_dist_dict.values())[0]) + ' corresponding to the house with index: ' +
      str(list(sorted_dist_dict.keys())[0]))

# perform 1-nearest neighbor regression
diff = features_train[::] - features_test[0]
print(sum(diff[-1]))


def dist_vect(x, y):
    return np.sqrt(np.sum((x-y)**2, axis=1))


# distances = dist_vect(features_test[0], features_train[::])
distances = dist_vect(features_test[2], features_train[::])
print(min(distances))
print(train_data[np.where(distances == min(distances))[0][0]]['price'])


# perform k-nearest neighboor
def knn(k, features_matrix, query_features_vector):
    knn_dist_vect = dist_vect(query_features_vector, features_matrix)
    return list(np.argsort(knn_dist_vect)[:k])


def predict_knn(k, features_matrix, output, query_features_vector):
    knn_indexes = knn(k, query_features_vector, features_matrix)
    return np.average(np.take(output, knn_indexes))


print(predict_knn(4, features_test[2], train_data['price'], features_train))


def predict_knn_set(k, features_matrix_data, output, features_matrix_query):
    prediction_set = np.empty(features_matrix_query.shape[0])

    for i in range(features_matrix_query.shape[0]):
        prediction_set[i] = predict_knn(k, features_matrix_query[i], output, features_matrix_data)

    return prediction_set


predictions_knn = predict_knn_set(10, features_train, train_data['price'], features_test[:10])
print(predictions_knn)

print(np.where(predictions_knn == min(predictions_knn))[0][0])

rss = []
k_list = range(1, 16)
for k in k_list:
    predictions = predict_knn_set(k, features_train, train_data['price'], features_validation)
    rss.append(sum((predictions - validation_data['price'])**2))

plt.plot(k_list, rss)
min_k = k_list[rss.index(min(rss))]

predictions_test_with_optimal_k = predict_knn_set(min_k, features_train, train_data['price'], features_test)
test_rss = sum((predictions_test_with_optimal_k - test_data['price'])**2)
