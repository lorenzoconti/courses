import turicreate as tc
import pandas as pd
import numpy as np

loans = tc.SFrame.read_csv('lending-club-data.csv')

# target (or label) column
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]

target = 'safe_loans'

# Extract the feature columns and target column
loans = loans[features + [target]]

categorical_variables = []
for feat_name, feat_type in zip(loans.column_names(), loans.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = pd.get_dummies(loans.to_dataframe()[feature], prefix=feature)
    loans = loans.remove_column(feature)

    for col in loans_data_one_hot_encoded.columns:
        loans[col] = loans_data_one_hot_encoded[col]

train_data, validation_data = loans.random_split(.8, seed=1)


def reached_minimum_node_size(data, min_node_size):

    if len(data) <= min_node_size:
        return True
    else:
        return False


def error_reduction(error_before_split, error_after_split):
    return error_before_split - error_after_split


def intermediate_node_num_mistakes(labels):
    if len(labels) == 0:
        return 0
    safe_loans_num = sum(labels == 1)
    risky_loans_num = sum(labels == -1)
    return min(safe_loans_num, risky_loans_num)


def best_splitting_feature(data, splitting_features, target):

    from math import inf

    target_values = data[target]
    best_feature = None
    best_error = inf

    num_data_points = float(len(data))

    for f in splitting_features:

        left_split = data[data[f] == 0]
        right_split = data[data[f] == 1]

        left_mistakes = intermediate_node_num_mistakes(left_split[target])
        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        error = (left_mistakes + right_mistakes) / num_data_points

        if error < best_error:
            best_error = error
            best_feature = f

    return best_feature


def create_leaf(target_values):

    leaf = {'splitting_feature': None, 'left': None, 'right': None, 'is_leaf': True}

    ones = len(target_values[target_values == 1])
    minus_ones = len(target_values[target_values == -1])

    if ones > minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    return leaf


def decision_tree_create(data, features, target, current_depth=0, max_depth=10, min_node_size=1,
                         min_error_reduction=0.0):

    remaining_features = features.copy()

    target_values = data[target]

    if intermediate_node_num_mistakes(target_values) == 0:
        return create_leaf(target_values)

    if len(remaining_features):
        return create_leaf(target_values)

    # early stopping condition: reached max depth limit
    if current_depth >= max_depth:
        return  create_leaf(target_values)

    # early stopping condition: reached minimum node size
    if reached_minimum_node_size(data, min_node_size):
        return create_leaf(target_values)

    splitting_feature = best_splitting_feature(data, features, target)

    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])

    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # early stopping condition: error reduction too little
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)

    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth+1, max_depth, min_node_size,
                                     min_error_reduction)

    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth,
                                      min_node_size, min_error_reduction)

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree
            }


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])


features = list(train_data.column_names())
features.remove('safe_loans')

decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=100,
                                     min_error_reduction=0.0)


def classify(tree, x):

    if tree['is_leaf']:
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if split_feature_value == 0:
            return classify(tree['left'], x)
        else:
            return classify(tree['right'], x)


print(classify(decision_tree, validation_data[0]))


def evaluate_classification_error(tree, data):

    prediction = data.apply(lambda x: classify(tree, x))

    return sum(data['safe_loans'] != prediction) / len(data)


print(evaluate_classification_error(decision_tree, validation_data))

# exploring the effect of max_depth

models = []
for d in [2, 6, 14]:
    tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=d, min_node_size=100,
                                min_error_reduction=0.0)
    models.append(tree)
    print('model with depth {} validation accuracy: {}'.format(d, evaluate_classification_error(tree, validation_data)))
    print('model with depth {} training accuracy: {}'.format(d, evaluate_classification_error(tree, train_data)))
    print('model with depth {} number of leaves: {}'.format(d, count_leaves(tree)))
    print('')

for e in [-1, 0, 5]:
    tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=100,
                                min_error_reduction=e)
    models.append(tree)
    print('model error reduction {} validation accuracy: {}'.format(d, evaluate_classification_error(tree, validation_data)))
    print('model error reduction {} training accuracy: {}'.format(d, evaluate_classification_error(tree, train_data)))
    print('model error reduction {} number of leaves: {}'.format(d, count_leaves(tree)))
    print('')

for size in [0, 2000, 5000]:
    tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=100,
                                min_error_reduction=-1)
    models.append(tree)
    print('model with node size {} validation accuracy: {}'.format(d, evaluate_classification_error(tree, validation_data)))
    print('model with node size {} training accuracy: {}'.format(d, evaluate_classification_error(tree, train_data)))
    print('model with node size {} number of leaves: {}'.format(d, count_leaves(tree)))
    print('')


