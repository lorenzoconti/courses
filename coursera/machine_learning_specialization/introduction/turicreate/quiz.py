import turicreate as tc

sf = tc.SFrame('people_wiki.sframe')

print(sf['name'].tail())

print(sf.filter_by('Harpdog Brown', 'name')['URI'])

print(sf.sort('text').head())cd
