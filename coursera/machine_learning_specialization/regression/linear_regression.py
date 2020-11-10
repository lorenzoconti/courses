import turicreate as tc
import matplotlib.pyplot as plt

sales = tc.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv')

crime_model = tc.linear_regression.create(sales, target='HousePrice', features=['CrimeRate'], validation_set=None)

plt.plot(sales['CrimeRate'], sales['HousePrice'], '.', sales['CrimeRate'], crime_model.predict(sales), 'r-')

# remove the center of the city
sales_noCC = sales[sales['MilesPhila'] != 0.0]
crime_model_noCC = tc.linear_regression.create(sales_noCC, target='HousePrice', features=['CrimeRate'],
                                               validation_set=None)

plt.plot(sales_noCC['CrimeRate'], crime_model_noCC.predict(sales_noCC), 'g-')

print(crime_model.__getattribute__('coefficients'))
print(crime_model_noCC.__getattribute__('coefficients'))

sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000]
crime_model_nohighend = tc.linear_regression.create(sales_nohighend, target='HousePrice', features=['CrimeRate'],
                                                    validation_set=None)
plt.plot(sales_nohighend['CrimeRate'], crime_model_nohighend.predict(sales_nohighend), 'y-')

print(crime_model_nohighend.__getattribute__('coefficients'))

# Assignment Section
train_data = tc.SFrame.read_csv('kc_house_train_data.csv')
test_data = tc.SFrame.read_csv('kc_house_test_data.csv')


def simple_linear_regression(input_features, output):
    sum_y = sum(output)
    sum_x = sum(input_features)
    sum_yx = sum(output*input_features)
    sum_xx = sum(input_features**2)
    n = len(output)
    slope = (sum_yx - ((sum_y*sum_x)/n))/(sum_xx - ((sum_x**2)/n))
    intercept = (sum_y/n) - slope*(sum_x/n)
    return slope, intercept


input_feature = train_data['sqft_living']
output = train_data['price']

sqft_slope, sqft_intercept = simple_linear_regression(input_feature, output)


def get_regression_predictions(input_feature, intercept, slope):
    return input_feature*slope + intercept


print('sqft prediction: {}'.format(get_regression_predictions(2650, sqft_intercept, sqft_slope)))


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    y_hat = slope*input_feature + intercept
    return sum((output - y_hat)**2)


print('rss sqft: {}'.format(get_residual_sum_of_squares(input_feature, output, sqft_intercept, sqft_slope)))


def inverse_regression_predictions(output, intercept, slope):
    return (output-intercept)/slope


print('inverse prediction: {}'.format(inverse_regression_predictions(800000, sqft_intercept, sqft_slope)))

# model with bedrooms
bed_slope, bed_intercept = simple_linear_regression(train_data['bedrooms'], train_data['price'])
bed_rss = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bed_intercept, bed_slope)

sqft_rss = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)




