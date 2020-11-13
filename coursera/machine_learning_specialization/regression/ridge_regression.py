import turicreate as tc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def polynomial_sframe(feature, degree):
    poly_sframe = tc.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe


sales = tc.SFrame('home_data.sframe')
sales = sales.sort(['sqft_living', 'price'])

l2_small_penalty = 1e-5

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
features = poly15_data.column_names()
poly15_data['price'] = sales['price']

model15 = tc.linear_regression.create(poly15_data, target='price', features=features, validation_set=None,
                                      l2_penalty=l2_small_penalty, verbose=False)

plt.plot(poly15_data['power_1'], poly15_data['price'], '.', poly15_data['power_1'], model15.predict(poly15_data), '-')

print(model15.__getattribute__('coefficients')[1])

sales_1, sales_2 = sales.random_split(0.5, seed=0)
set_1, set_2 = sales_1.random_split(0.5, seed=0)
set_3, set_4 = sales_2.random_split(0.5, seed=0)


def create_polynomial(dataset, feature, output, degree):
    target = dataset[output]
    result = polynomial_sframe(dataset[feature], degree)
    result[output] = target
    return result


set_1 = create_polynomial(set_1, 'sqft_living', 'price', 15)
set_2 = create_polynomial(set_2, 'sqft_living', 'price', 15)
set_3 = create_polynomial(set_3, 'sqft_living', 'price', 15)
set_4 = create_polynomial(set_4, 'sqft_living', 'price', 15)

models = []
for s in [set_1, set_2, set_3, set_4]:
    models.append(tc.linear_regression.create(s, target='price', features=features, validation_set=None,
                                              l2_penalty=l2_small_penalty, verbose=False))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(set_1['power_1'], set_1['price'], '.', set_1['power_1'], models[0].predict(set_1))
axs[0, 1].plot(set_2['power_1'], set_2['price'], '.', set_2['power_1'], models[1].predict(set_2))
axs[1, 0].plot(set_3['power_1'], set_3['price'], '.', set_3['power_1'], models[2].predict(set_3))
axs[1, 1].plot(set_4['power_1'], set_4['price'], '.', set_4['power_1'], models[3].predict(set_4))

for i in range(len(models)):
    print('{} : {}'.format(i, models[i].__getattribute__('coefficients')[1]))


l2_penalty = 1e5

models = []
for s in [set_1, set_2, set_3, set_4]:
    models.append(tc.linear_regression.create(s, target='price', features=features, validation_set=None,
                                              l2_penalty=l2_penalty, verbose=False))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(set_1['power_1'], set_1['price'], '.', set_1['power_1'], models[0].predict(set_1))
axs[0, 1].plot(set_2['power_1'], set_2['price'], '.', set_2['power_1'], models[1].predict(set_2))
axs[1, 0].plot(set_3['power_1'], set_3['price'], '.', set_3['power_1'], models[2].predict(set_3))
axs[1, 1].plot(set_4['power_1'], set_4['price'], '.', set_4['power_1'], models[3].predict(set_4))

for i in range(len(models)):
    print('{} : {}'.format(i, models[i].__getattribute__('coefficients')[1]))

# k-fold cross validation

dtype_dict = {
    'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int,
    'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
    'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
    'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int
}

train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)


def k_fold_cross_validation(k, l2, data, target, degree):

    n = len(data)
    errors = []
    features = ['power_' + str(i) for i in range(1, degree+1)]

    for i in range(k):
        start = (n * i) / k
        end = (n * (i + 1)) / k - 1

        train_set = data[0:int(start)].append(data[int(end+1):n])
        validation_set = data[int(start):int(end+1)]
        train_set = create_polynomial(train_set, 'sqft_living', 'price', degree)
        validation_set = create_polynomial(validation_set, 'sqft_living', 'price', degree)
        model = tc.linear_regression.create(train_set, features=features, target=target, l2_penalty=l2,
                                            validation_set=None, verbose=False)
        predictions = model.predict(validation_set)
        errors.append(sum((validation_set[target]-predictions)**2))

    return sum(errors)/len(errors)


errors = []
l2s = np.logspace(1, 7, num=13)
for l2 in l2s:
    errors.append(k_fold_cross_validation(10, l2, train_valid_shuffled, 'price', 15))

plt.figure()
plt.xscale('log')
plt.plot(l2s, errors)

print('min: {} for l2: {}'.format(min(errors), l2s[errors.index(min(errors))]))


train_set, test_set = sales.random_split(.9, seed=0)
model = tc.linear_regression.create(create_polynomial(train_set, 'sqft_living', 'price', 15), features=features,
                                    target='price', l2_penalty=l2s[errors.index(min(errors))], validation_set=None,
                                    verbose=False)

predictions = model.predict(create_polynomial(test_set, 'sqft_living', 'price', 15))

print(sum((test_set['price'] - predictions)**2))

plt.figure()
plt.plot(test['sqft_living'], test['price'], '.', test_set['sqft_living'], predictions, '-')
