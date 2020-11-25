import turicreate as tc
from math import log, exp
import matplotlib.pyplot as plt

loans = tc.SFrame.read_csv('lending-club-data.csv')

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]

target = 'safe_loans'
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed=1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print('Percentage of safe loans                 : {}'.format(len(safe_loans) / float(len(loans_data))))
print('Percentage of risky loans                : {}'.format(len(risky_loans) / float(len(loans_data))))
print('Total number of loans in our new dataset : {}'.format(len(loans_data)))

# one hot encoding
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data = loans_data.remove_column(feature)
    loans_data = loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names()
features.remove('safe_loans')

train_data, test_data = loans_data.random_split(.8, seed=1)


# weighted decision trees
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # sum of weights of all entries with label +1
    total_weights_positive = sum(data_weights[labels_in_node == 1])

    # weight of mistakes for predicting all -1's is equal to the sum above
    weighted_mistakes_all_negative = sum(data_weights[labels_in_node == +1])

    weighted_mistakes_all_positive = total_weights_negative= sum(data_weights[labels_in_node == -1])

    return min(weighted_mistakes_all_negative, weighted_mistakes_all_positive), 1 \
        if  weighted_mistakes_all_positive <= weighted_mistakes_all_negative else -1


def best_splitting_feature(data, features, target, data_weights):

    best_feature = None
    best_error = float('+inf')
    num_points = float(len(data))

    for feature in features:

        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]

        left_data_weights = data_weights[data[feature] == 0]
        right_data_weights = data_weights[data[feature] == 1]

        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)

        error = (left_weighted_mistakes + right_weighted_mistakes) / float(sum(data_weights))

        if error < best_error:
            best_error = error
            best_feature = feature

    return best_feature


def create_leaf(target_values, data_weights):
    leaf = {'splitting_feature': None, 'is_leaf': True}
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    return leaf


def weighted_decision_tree_create(data, features, target, data_weights, current_depth=1, max_depth=10):

    remaining_features = features.copy()
    target_values = data[target]

    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        return create_leaf(target_values, data_weights)

    if len(remaining_features) == 0:
        return create_leaf(target_values, data_weights)

    if current_depth > max_depth:
        return create_leaf(target_values, data_weights)

    splitting_feature = best_splitting_feature(data, remaining_features, target, data_weights)
    remaining_features.remove(splitting_feature)

    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]

    # create a leaf node if the split is perfect
    if len(left_split) == len(data):
        return create_leaf(left_split[target], data_weights)

    if len(right_split) == len(data):
        return create_leaf(right_split[target], data_weights)

    left_tree = weighted_decision_tree_create(left_split, remaining_features, target, left_data_weights,
                                              current_depth+1, max_depth)

    right_tree = weighted_decision_tree_create(right_split, remaining_features, target, right_data_weights,
                                               current_depth+1, max_depth)

    return {'is_leaf': False, 'prediction': None, 'splitting_feature': splitting_feature, 'left': left_tree,
            'right': right_tree}


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def classify(tree, x):

    if tree['is_leaf']:
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if split_feature_value == 0:
            return classify(tree['left'], x)
        else:
            return classify(tree['right'], x)


def evaluate_classification_error(tree, data):
    predictions = data.apply(lambda x: classify(tree, x))
    return sum(predictions != data[target]) / float(len(data))


def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):

    alpha = tc.SArray([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]

    for t in range(num_tree_stumps):

        print('adaboost iteration {}'.format(t))

        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)

        predictions = data.apply(lambda x: classify(tree_stump, x))

        is_correct = predictions == target_values
        is_wrong = predictions != target_values

        weighted_error = sum(alpha*is_wrong)/sum(alpha)

        weight = log((1 - weighted_error)/weighted_error)*0.5
        weights.append(weight)

        adjustment = is_correct.apply(lambda correct: exp(-weight) if correct else exp(weight))

        alpha = alpha * adjustment
        alpha = alpha/sum(alpha)

    return weights, tree_stumps


stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=10)


def print_stump(tree, name='root'):
    split_name = tree['splitting_feature']
    if split_name is None:
        print("(leaf, label: {})".format(tree['prediction']))
        return None

    print('                       {}'.format(name))
    print('         |---------------|----------------|')
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('    ({})                         ({})'.format(('leaf, label: ' + str(tree['left']['prediction'])
                                                         if tree['left']['is_leaf'] else 'subtree'),
                                                         ('leaf, label: ' + str(tree['right']['prediction'])
                                                          if tree['right']['is_leaf'] else 'subtree')))


print_stump(tree_stumps[0])


def predict_adaboost(stump_weights, tree_stumps, data):

    scores = tc.SArray([0.]*len(data))

    for i, tree_stump in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify(tree_stump, x)*stump_weights[i])
        scores += predictions

    return scores.apply(lambda s: 1 if s > 0 else -1)


predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
accuracy = tc.evaluation.accuracy(test_data[target], predictions)
print('accuracy: {}'.format(accuracy))

# 62.37 without stump_weigths multiplication in predict adaboost
# 62.03 with

stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=30)

errors = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    error = 1 - tc.evaluation.accuracy(train_data[target], predictions)
    errors.append(error)

plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1, 31), errors, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})

errors = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], test_data)
    error = 1 - tc.evaluation.accuracy(test_data[target], predictions)
    errors.append(error)

plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1, 31), errors, '-', linewidth=4.0, label='Test error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size': 15})
plt.rcParams.update({'font.size': 16})

