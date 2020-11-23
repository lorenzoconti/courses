import turicreate as tc
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from os import system

loans = tc.SFrame.read_csv('lending-club-data.csv')

# target (or label) column
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

print(loans['safe_loans'].value_counts())
print(len(loans[loans['safe_loans'] == 1])/len(loans))
print(len(loans[loans['safe_loans'] == -1])/len(loans))

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

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

train_data, validation_data = loans_data.random_split(.8, seed=1)

train_target = train_data['safe_loans'].to_numpy()
train_x = train_data.remove_column('safe_loans').to_numpy()

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model = decision_tree_model.fit(train_x, train_target)

small_model = DecisionTreeClassifier(max_depth=2)
small_model = small_model.fit(train_x, train_target)

tree.export_graphviz(small_model, out_file='tree.dot')

# system('dot -Tpng tree.dot -o tree-png ')

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_safe_loans = validation_safe_loans[0:2]
sample_validation_risky_loans = validation_risky_loans[0:2]

sample_validation_data = sample_validation_safe_loans.append(sample_validation_risky_loans)

sample_validation_data_target = sample_validation_data['safe_loans'].to_numpy()
sample_validation_data_x = sample_validation_data.remove_column('safe_loans').to_numpy()


print(decision_tree_model.predict(sample_validation_data_x))
print(decision_tree_model.predict_proba(sample_validation_data_x))

print(small_model.predict(sample_validation_data_x))
print(small_model.predict_proba(sample_validation_data_x))

print(small_model.score(train_x, train_target))
print(decision_tree_model.score(train_x, train_target))

validation_target = validation_data['safe_loans'].to_numpy()
validation_x = validation_data.remove_column('safe_loans').to_numpy()

print(small_model.score(validation_x, validation_target))
print(decision_tree_model.score(validation_x, validation_target))

# evaluating a bigger model
big_model = DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_x, train_target)

print(big_model.score(train_x, train_target))
print(big_model.score(validation_x, validation_target))

predictions = decision_tree_model.predict(validation_x)
false_positive = sum((predictions == 1) * (validation_target == -1))
false_negaitves = sum((predictions == -1) * (validation_target == 1))

correct_predictions = sum(predictions == validation_target)

print('total cost: {}'.format(10000*false_negaitves + 20000*false_positive))