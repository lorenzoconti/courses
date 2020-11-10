import turicreate as tc
from math import log

train_data = tc.SFrame.read_csv('kc_house_train_data.csv')
test_data = tc.SFrame.read_csv('kc_house_test_data.csv')


def transform_feature(data, feature, suffix, func):
    ft = feature + '_' + suffix
    data[ft] = data[feature].apply(func)


transform_feature(train_data, 'bedrooms', 'squared', lambda x: x**2)
transform_feature(test_data, 'bedrooms', 'squared', lambda x: x**2)

transform_feature(train_data, 'sqft_living', 'log', lambda x: log(x))
transform_feature(test_data, 'sqft_living', 'log', lambda x: log(x))

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

print(test_data['bedrooms_squared'].mean())
print(test_data['bed_bath_rooms'].mean())
print(test_data['sqft_living_log'].mean())
print(test_data['lat_plus_long'].mean())

m1_ft = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
m2_ft = m1_ft + ['bed_bath_rooms']
m3_ft = m2_ft + ['bedrooms_squared', 'sqft_living_log', 'lat_plus_long']

m1 = tc.linear_regression.create(train_data, target='price', features=m1_ft, validation_set=None)
m2 = tc.linear_regression.create(train_data, target='price', features=m2_ft, validation_set=None)
m3 = tc.linear_regression.create(train_data, target='price', features=m3_ft, validation_set=None)

print('m1 coefficients: {}'.format(m1.__getattribute__('coefficients')))
print('m2 coefficients: {}'.format(m2.__getattribute__('coefficients')))
print('m3 coefficients: {}'.format(m3.__getattribute__('coefficients')))


def rss(model, data, target):
    return sum((data[target] - model.predict(data)) ** 2)


print('model 1 rss: {}'.format(rss(m1, train_data, 'price')))
print('model 2 rss: {}'.format(rss(m2, train_data, 'price')))
print('model 3 rss: {}'.format(rss(m3, train_data, 'price')))

print('model 1 rss: {}'.format(rss(m1, test_data, 'price')))
print('model 2 rss: {}'.format(rss(m2, test_data, 'price')))
print('model 3 rss: {}'.format(rss(m3, test_data, 'price')))

