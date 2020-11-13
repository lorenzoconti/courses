import turicreate as tc
import matplotlib.pyplot as plt


def polynomial_sframe(feature, degree):

    if degree < 1:
        return

    poly_sframe = tc.SFrame()
    poly_sframe['power_1'] = feature

    for power in range(2,degree+1):
        name = 'power_' + str(power)
        poly_sframe[name] = feature.apply(lambda x: x**power)

    return poly_sframe


sales = tc.SFrame.read_csv('kc_house_data.csv')
sales = sales.sort(['sqft_living', 'price'])

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']
model1 = tc.linear_regression.create(poly1_data, target='price', features=['power_1'], validation_set=None)
plt.plot(poly1_data['power_1'], poly1_data['price'], '.', poly1_data['power_1'], model1.predict(poly1_data))

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']
model2 = tc.linear_regression.create(poly2_data, target='price', features=['power_1', 'power_2'],
                                     validation_set=None)
plt.plot(poly2_data['power_1'], model2.predict(poly2_data))

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']
model3 = tc.linear_regression.create(poly3_data, target='price', features=['power_1', 'power_2', 'power_3'],
                                     validation_set=None)
plt.plot(poly3_data['power_1'], model3.predict(poly3_data))

features = []
for i in range(1, 16):
    features.append('power_'+str(i))

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']
model15 = tc.linear_regression.create(poly15_data, target='price', features=features, validation_set=None)
plt.plot(poly15_data['power_1'], poly15_data['price'], '.', poly15_data['power_1'], model15.predict(poly15_data))

# splitting the dataset
sales_1, sales_2 = sales.random_split(0.5, seed=0)
set_1, set_2 = sales_1.random_split(0.5, seed=0)
set_3, set_4 = sales_2.random_split(0.5, seed=0)


def create_polynomial(dataset, feature, output, degree):
    dataset = dataset.sort(['sqft_living', 'price'])
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
    models.append(tc.linear_regression.create(s, target='price', features=features, validation_set=None))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(set_1['power_1'], set_1['price'], '.', set_1['power_1'], models[0].predict(set_1))
axs[0, 1].plot(set_2['power_1'], set_2['price'], '.', set_2['power_1'], models[1].predict(set_2))
axs[1, 0].plot(set_3['power_1'], set_3['price'], '.', set_3['power_1'], models[2].predict(set_3))
axs[1, 1].plot(set_4['power_1'], set_4['price'], '.', set_4['power_1'], models[4].predict(set_4))

for m in models:
    print(m.__getattribute__('coefficients')[15])

# selecting the best polynomial model with cross validation
train_data, test_data = sales.random_split(.9, seed=1)
train_data, validation_data = train_data.random_split(.5, seed=1)


def rss(model, data, target):
    return sum((data[target] - model.predict(data)) ** 2)


# array of rss
rsss = []
models = []
for degree in range(1, 15+1):

    data = create_polynomial(train_data, 'sqft_living', 'price', degree)
    validation = create_polynomial(validation_data, 'sqft_living', 'price', degree)
    features = ['power_' + str(i) for i in range(1, degree+1)]
    model = tc.linear_regression.create(data, features=features, target='price', validation_set=None, verbose=False)
    models.append(model)
    rsss.append(rss(model, validation, 'price'))


plt.figure()
plt.plot(range(1, 16), rsss)

min_rss = rsss.index(min(rsss))

test = create_polynomial(test_data, 'sqft_living', 'price', min_rss)
poly_frame = create_polynomial(train_data, 'sqft_living', 'price', min_rss)
features = ['power_' + str(i) for i in range(1, min_rss+1)]
best_model = tc.linear_regression.create(poly_frame, target='price', features=features, validation_set=None )
print(rss(best_model, test_data, 'price'))
