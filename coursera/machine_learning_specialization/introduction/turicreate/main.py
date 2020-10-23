import turicreate as tc

# SFrame
sf = tc.SFrame('people.csv')

# print the first few lines
print(sf)
print(sf.head())

# print the last few lines
print(sf.tail())

# visualization
# sf.show()
sf['age'].show()

# inspect columns
print(sf['Country'])
print(sf['Age'].max())
print(sf['Age'].mean())

sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']


# advanced transformation using apply


def transform_country(country):
    if country == 'USA':
        return 'United States'
    else:
        return country


sf['Country'] = sf['Country'].apply(transform_country)






