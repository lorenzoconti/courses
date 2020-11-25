import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt

products = tc.SFrame.read_csv('amazon_baby.csv')


def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


review_clean = products['review'].apply(remove_punctuation)

products['word_count'] = tc.text_analytics.count_words(review_clean)

products = products[products['rating'] != 3]

products['sentiment'] = products['rating'].apply(lambda r: +1 if r > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

model = tc.logistic_classifier.create(train_data, target='sentiment', features=['word_count'], validation_set=None)

accuracy = model.evaluate(test_data, metric='accuracy')['accuracy']

baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)

confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']

# precision and recall
precision = model.evaluate(test_data, metric='precision')['precision']
recall = model.evaluate(test_data, metric='recall')['recall']


def apply_threshold(probabilities, threshold):
    return probabilities.apply(lambda x: 1 if x >= threshold else -1)


probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

print('number of predicted as positive (threshold 0.5): {}'.format(sum(predictions_with_default_threshold)))
print('number of predicted as positive (threshold 0.9): {}'.format(sum(predictions_with_high_threshold)))

precision_with_default_threshold = tc.evaluation.precision(test_data['sentiment'], predictions_with_default_threshold)
recall_with_default_threshold = tc.evaluation.recall(test_data['sentiment'], predictions_with_default_threshold)

precision_with_high_threshold = tc.evaluation.precision(test_data['sentiment'], predictions_with_high_threshold)
recall_with_high_threshold = tc.evaluation.recall(test_data['sentiment'], predictions_with_high_threshold)

print('precision/recall 0.5: {} / {}'.format(precision_with_default_threshold, recall_with_default_threshold))
print('precision/recall 0.9: {} / {}'.format(precision_with_high_threshold, recall_with_high_threshold))

thresholds = np.linspace(0.5, 1, num=100)
precisions = []
recalls = []

probabilities = model.predict(test_data, output_type='probability')
for threshold in thresholds:

    predictions = apply_threshold(probabilities, threshold)
    precisions.append(tc.evaluation.precision(test_data['sentiment'], predictions))
    recalls.append(tc.evaluation.recall(test_data['sentiment'], predictions))


def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})


plot_pr_curve(precisions, recalls, 'precision recall curve')

# what is the smallest threshold value that achieves a precision of 96.5 or higher?
print(min([x for i, x in enumerate(thresholds) if precisions[i] > 0.965]))

predictions_with_high_threshold = apply_threshold(probabilities, 0.98)
print(tc.evaluation.confusion_matrix(test_data['sentiment'], predictions_with_high_threshold))

# evaluating specific search items
baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]

probabilities = model.predict(baby_reviews, output_type='probability')
thresholds = np.linspace(0.5, 1, num=100)
precisions = []
recalls = []
for threshold in thresholds:

    predictions = apply_threshold(probabilities, threshold)
    precisions.append(tc.evaluation.precision(baby_reviews['sentiment'], predictions))
    recalls.append(tc.evaluation.recall(baby_reviews['sentiment'], predictions))

print(min([x for i, x in enumerate(thresholds) if precisions[i] > 0.965]))

plot_pr_curve(precisions, recalls, 'precision recall (babby)')