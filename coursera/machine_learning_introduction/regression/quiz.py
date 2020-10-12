import turicreate as tc
import turicreate.aggregate as agg

sales = tc.SFrame.read_csv('data/home_data.csv')
train_data, test_data = sales.random_split(.8, seed=0)

# highest_price = sales['price'].max()
# zipcode = sales[sales['price'] == highest_price]['zipcode']
# neighborhood = sales[sales['zipcode'] == zipcode[0]]
# avg = neighborhood['price'].mean()

stats = sales.groupby('zipcode', operations={'mean': agg.MEAN('price')})
zipcode = stats[stats['mean'] == stats['mean'].max()]['zipcode'][0]
print(stats['mean'].max())

# filtered = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] < 4000)]
filtered = sales[sales['sqft_living'].apply(lambda sqft: 2000 < sqft < 4000)]
print('fraction: {f:.2}'.format(f=len(filtered)/len(sales)))

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition',
    'grade', 'waterfront', 'view', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
    'sqft_living15', 'sqft_lot15'
]

model = tc.linear_regression.create(train_data, features=features, target='price', validation_set=None)
advanced_model = tc.linear_regression.create(train_data, features=advanced_features, target='price', validation_set=None)

model_result = model.evaluate(dataset=test_data)
advanced_result = advanced_model.evaluate(dataset=test_data)


def rss(rmse, n):
    return n*(rmse**2)


print('rss: {m:.3f} {advm:.3f}'.format(
    m=rss(model_result['rmse'], len(test_data)),
    advm=rss(advanced_result['rmse'], len(test_data))))

print('rmse: {m:.1f} {advm:.1f}'.format(m=model_result['rmse'], advm=advanced_result['rmse']))
print('rmse difference: {diff:.1f}'.format(diff=model_result['rmse']-advanced_result['rmse']))
