import turicreate as tc
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# load some house sales data

sales = tc.SFrame('machine_learning_introduction/data/home_data.sframe')

print(sales)

tc.visualization.scatter(x=sales['sqft_living'], y=sales['price']).show()

# create a simple regression model of square feet living to price
# .split the dataset: 8 train and .2 test

train_data, test_data = sales.random_split(.8, seed=0)

# build the regression model

sqft_model = tc.linear_regression.create(dataset=train_data, target='price', features=['sqft_living'])

# evaluate the model

print(test_data['price'].mean())

sqft_model.evaluate(dataset=test_data)

# visualize results

plt.plot(test_data['sqft_living'], test_data['price'], '.', test_data['sqft_living'], sqft_model.predict(test_data), '-')
plt.show()

print(sqft_model.__getattribute__('coefficients'))

# explore other features in the data

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

sales[features].show()

tc.visualization.box_plot(x=sales['zipcode'], y=sales['price']).show()

# build a regression model with more features

featured_model = tc.linear_regression.create(train_data, target='price', features=features)

print(sqft_model.evaluate(test_data))
print(featured_model.evaluate(test_data))

# apply learned models to predict prices of three houses

house1 = sales[sales['id'] == '5309101200']

# on notebook
# <img src="house-5309101200.jpeg">

print('real: {} , predicted by sqft_model: {}, predicted by featured model: {}'.format(house1['price'],
                                                                                       sqft_model.predict(house1),
                                                                                       featured_model.predict(house1)))

house2 = sales[sales['id'] == '1925069082']

# on notebook
# <img src="house-1925069082.jpg>

print(sqft_model.predict(house2))
print(featured_model.predict(house2))

# model.predict(tc.SFrames(dictionary))

