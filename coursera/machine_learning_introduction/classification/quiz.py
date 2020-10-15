import turicreate as tc

reviews = tc.SFrame.read_csv('../data/amazon_baby.csv')

reviews = reviews[reviews['rating'] != 3]

reviews['sentiment'] = reviews['rating'] >= 4

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible',
                  'bad', 'terrible', 'awful', 'wow', 'hate']

reviews['word_count'] = tc.text_analytics.count_words(reviews['review'])
selected_words_count = {}

for word in selected_words:
    reviews[str(word)] = reviews['word_count'].apply(lambda row: row[word] if word in row else 0)
    selected_words_count[word] = reviews[word].sum()

sorted_selected_words = sorted(selected_words_count.items(), key=lambda x: x[1], reverse=True)

train_data, test_data = reviews.random_split(.8, seed=0)

selected_words_model = tc.logistic_classifier.create(dataset=train_data, features=selected_words,
                                                     validation_set=test_data,
                                                     target='sentiment')

coefficients = selected_words_model.__getattribute__('coefficients')
coefficients = coefficients.sort('value')
evaluation_results = selected_words_model.evaluate(test_data)

diaper_champ_reviews = reviews[reviews['name'] == 'Baby Trend Diaper Champ']

sentiment_model = tc.logistic_classifier.create(dataset=train_data, target='sentiment', features=['word_count'],
                                                validation_set=test_data)

evaluation_results_sentiment = sentiment_model.evaluate(test_data)

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)\

result = selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')

