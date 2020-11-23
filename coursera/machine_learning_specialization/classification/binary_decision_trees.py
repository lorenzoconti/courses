import turicreate as tc
import pandas as pd
import numpy as np

loans = tc.SFrame.read_csv('lending-club-data.csv')

# target (or label) column
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]

target = 'safe_loans'

# Extract the feature columns and target column
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print("Number of safe loans  : " + str(len(safe_loans_raw)))
print("Number of risky loans : " + str(len(risky_loans_raw)))

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# one hot encoding
categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = pd.get_dummies(loans_data.to_dataframe()[feature], prefix=feature)
    loans_data = loans_data.remove_column(feature)

    for col in loans_data_one_hot_encoded.columns:
        loans_data[col] = loans_data_one_hot_encoded[col]

train_data, test_data = loans_data.random_split(.8, seed=1)


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


def decision_tree_create(data, current_features, target, current_depth=0, max_depth=10):

    # make a copy of the features
    remaining_features = current_features.copy()

    target_values = data[target]

    # stopping condition
    if intermediate_node_num_mistakes(target_values) == 0:
        return create_leaf(target_values)

    # stopping condition
    if len(remaining_features) == 0:
        return create_leaf(target_values)

    # stopping condition
    if current_depth >= max_depth:
        return create_leaf(target_values)

    splitting_feature = best_splitting_feature(data, remaining_features, target)

    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)

    if len(left_split) == len(data):
        return create_leaf(left_split[target])

    if len(right_split) == len(data):
        return create_leaf(right_split[target])

    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth+1, max_depth)
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth+1, max_depth)

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree
            }


input_features = list(train_data.column_names())
input_features.remove('safe_loans')

decision_tree = decision_tree_create(train_data, input_features, 'safe_loans', current_depth=0, max_depth=6)


def classify(tree, x):

    if tree['is_leaf']:
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if split_feature_value == 0:
            return classify(tree['left'], x)
        else:
            return classify(tree['right'], x)


print(classify(decision_tree, train_data[0]))


def evaluate_classification_error(tree, data):

    prediction = data.apply(lambda x: classify(tree, x))

    return sum(data['safe_loans'] != prediction) / len(data)


print(evaluate_classification_error(decision_tree, test_data))


def print_stump(tree, name='root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
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

    print_stump(decision_tree)

    print_stump(decision_tree['left'], decision_tree['splitting_feature'])

