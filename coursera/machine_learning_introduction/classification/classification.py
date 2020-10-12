# Analyzing product sentiment
import turicreate as tc
import matplotlib.pyplot as plt


products = tc.SFrame('../data/amazon_baby/amazon_baby.sframe')

print(products.head())

# build the word count vector for each review
products['word_count'] = tc.text_analytics.count_words(products['review'])

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
print(giraffe_reviews)

# tc.visualization.histogram(giraffe_reviews['rating']).show()
# tc.visualization.histogram(products['rating']).show()

# define what is a positive and negative sentiment
# ignore all three stars reviews
products = products[products['rating'] != 3]

# positive sentiment: 4 or 5 stars
# negative sentiment: 1 or 2 stars

products['sentiment'] = products['rating'] >= 4

# train sentiment classifier
train_data, test_data = products.random_split(.8, seed=0)

sentiment_model = tc.logistic_classifier.create(dataset=train_data, target='sentiment', features=['word_count'],
                                                validation_set=test_data)
# evaluate the sentiment model
evaluation_results = sentiment_model.evaluate(test_data, metric='roc_curve')

# missing the implementation of model.show(view='Evaluate') with turicreate
plt.plot(evaluation_results['roc_curve']['fpr'], evaluation_results['roc_curve']['tpr'])

# apply the model to understand sentiment for giraffe
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')

# sort the reviews based on the predicted sentiment and explore
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
print(giraffe_reviews[0]['review'])
print(giraffe_reviews[-1]['review'])