import turicreate as tc
from math import log, sqrt
import numpy as np
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv('kc_house_data.csv')

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors']*sales['floors']

features = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 'sqft_living_sqrt', 'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']

model = tc.linear_regression.create(sales, target='price', features=features, validation_set=None, l2_penalty=0,
                                    l1_penalty=1e10)

non_zero_features = model.coefficients["value"] > 0
model.coefficients[non_zero_features].print_rows(num_rows=20)

# selecting an L1 penalty
train_data, test_data = sales.random_split(.9, seed=1)
train_data, validation_data = train_data.random_split(.5, seed=1)


def rss(model, data, output):
    predictions = model.predict(data)
    residuals = predictions - output
    return sum(residuals**2)


# dict implementation
# rsss = {}
rsss = []
l1_penalties = np.logspace(1, 7, num=13)
for l1 in l1_penalties:
    # rsss[l1] = ...
    rsss.append(rss(tc.linear_regression.create(train_data, target='price', features=features, validation_set=None,
                                                l2_penalty=0, l1_penalty=l1, verbose=False),
                    validation_data, validation_data['price']))

# min_rss = min(rsss.values())
# best_l1 = [key for key, value in rsss.items() if value == min_rss]
best_l1 = l1_penalties[rsss.index(min(rsss))]
print(min(rsss))
plt.plot(np.logspace(1, 7, num=13), rsss)

best_model = tc.linear_regression.create(train_data, target='price', features=features, validation_set=None,
                                         l2_penalty=0, l1_penalty=best_l1, verbose=False)

print("best model's rss: {}".format(rss(best_model, test_data, test_data['price'])))
non_zero_features = best_model.coefficients["value"] > 0
best_model.coefficients[non_zero_features].print_rows(num_rows=20)
print(best_model.coefficients['value'].nnz())

max_nonzeros = 7
l1_penalties = np.logspace(8, 10, num=20)

l1_nnz = []
rsss = []
for l1 in l1_penalties:
    model = tc.linear_regression.create(train_data, features=features, target='price', validation_set=None,
                                        l2_penalty=0, l1_penalty=l1, verbose=False)
    l1_nnz.append(model.coefficients['value'].nnz())
    rsss.append(rss(model, validation_data, validation_data['price']))

lower_end_filter = [value > 7 for value in l1_nnz]
lower_l2 = np.array(l1_penalties)[lower_end_filter][-1]
upper_end_filter = [value < 7 for value in l1_nnz]
upper_l2 = np.array(l1_penalties)[upper_end_filter][0]

l1_penalties = np.linspace(lower_l2, upper_l2, 20)
rsss = []
l1_nnz = []
for l1 in l1_penalties:
    model = tc.linear_regression.create(train_data, features=features, target='price', validation_set=None,
                                        l2_penalty=0, l1_penalty=l1, verbose=False)
    l1_nnz.append(model.coefficients['value'].nnz())
    rsss.append(rss(model, validation_data, validation_data['price']))

exactly_seven_filter = [value == 7 for value in l1_nnz]
exaclty_seven = min(np.array(rsss)[exactly_seven_filter])


