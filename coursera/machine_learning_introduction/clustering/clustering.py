import turicreate as tc

people = tc.SFrame('machine_learning_introduction/data/people_wiki.sframe')

obama = people[people['name'] == 'Barack Obama']
clooney = people[people['name'] == 'George Clooney']

# count the words
obama['word_count'] = tc.text_analytics.count_words(obama['text'])

# sort the words for the obama article
obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name= ['word', 'count'])

obama_word_count_table.sort('count', ascending=False)

# compute the TF-IDF for the corpus
people['word_count'] = tc.text_analytics.count_words(people['text'])
print(people.head())

tfidf = tc.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf

# examine the TF-IDF for the obama article
obama = people[people['name'] == 'Barack Obama']
obama_tfidf = obama[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf']).sort('tfidf', ascending=False)

# manually compute distances between a few people
clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']

# is obama closer to clinton than to beckham?
tc.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])
tc.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])

# build a nearest neighbot model for document retrieval
knn_model = tc.nearest_neighbors.create(people, features=['tfidf'], label='name')

# applying the nearest neighbor model for retrieval
# who is closest to obama?
knn_model.query(obama)

# other examples of document retrieval
swift = people[people['name'] == 'Taylor Swift']
knn_model.query(swift)

jolie = people[people['name'] == 'Angelina Jolie']
knn_model.query(jolie)

# assigmnent section

elton_john = people[people['name'] == 'Elton John']

elton_john_word_count = elton_john[['word_count']].stack('word_count', new_column_name=['word', 'count'])\
    .sort('count', ascending=False)

elton_john_tfidf = elton_john[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf'])\
    .sort('tfidf', ascending=False)

victoria_beckham = people[people['name'] == 'Victoria Beckham']
paul_mccartney = people[people['name'] == 'Paul McCartney']

tc.distances.cosine(elton_john['tfidf'][0], victoria_beckham['tfidf'][0])
tc.distances.cosine(elton_john['tfidf'][0], paul_mccartney['tfidf'][0])

knn_tfidf_model = tc.nearest_neighbors.create(people, features=['tfidf'], label='name', distance='cosine')
knn_wordcount_model = tc.nearest_neighbors.create(people, features=['word_count'], label='name', distance='cosine')

knn_wordcount_model.query(elton_john)
knn_tfidf_model.query(elton_john)

knn_wordcount_model.query(victoria_beckham)
knn_tfidf_model.query(victoria_beckham)