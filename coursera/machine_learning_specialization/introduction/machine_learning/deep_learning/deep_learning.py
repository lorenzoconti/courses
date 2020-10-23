import turicreate as tc

# load dataset
image_train = tc.SFrame('../data/image_train_data/')
image_test = tc.SFrame('../data/image_test_data/')

image_train[0:2]['image'][0].show()

# train a classifier on the raw image pixels
raw_pixel_model = tc.logistic_classifier.create(image_train, target='label', features=['image_array'])

# make a prediction with the simple model based on raw pixels
print(image_test[0:3]['label'])
# cat, automobile, cat

raw_pixel_model.predict(image_test[0:3])
# bird, cat, bird

# evaluating raw pixel model on test data
evaluation_raw = raw_pixel_model.evaluate(image_test)
# 48.1% accuracy

# can we improve the model using deep features?

# deep_learning_model = tc.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)

# given the deep feature let's train a classifier
deep_features_model = tc.logistic_classifier.create(image_train, target='label', features=['deep_features'])

deep_features_model.predict(image_test[0:3])
# cat, automobile, cat

evaluation_deep_features = deep_features_model.evaluate(image_test)
# 79.0% accuracy

# train a nearest neighbors model for retrieving images using deep features
knn_model = tc.nearest_neighbors.create(image_train, features=['deep_features'], label='id')

# use image retrieval model with deep features to find similar images
cat = image_train[18:19]
cat['image'][0].show()


def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'], 'id')


cat_neighbors = get_images_from_ids(knn_model.query(cat))
for cat in cat_neighbors:
    print(cat['label'])

car = image_train[8:90]
car['image'][0].show()
get_images_from_ids(knn_model.query(car))['image'][0:1].show()


def show_neighbors(i):
    get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'][1].show()


image_train[24:25]['image'][0].show()
show_neighbors(24)


# deep features for image retrieval assignment
sketch = tc.Sketch(image_train['label'])

cat_data = image_train[image_train['label'] == 'cat']
dog_data = image_train[image_train['label'] == 'dog']
car_data = image_train[image_train['label'] == 'automobile']
bird_data = image_train[image_train['label'] == 'bird']

cat_model = tc.nearest_neighbors.create(cat_data, features=['deep_features'], label='id')
dog_model = tc.nearest_neighbors.create(dog_data, features=['deep_features'], label='id')
car_model = tc.nearest_neighbors.create(car_data, features=['deep_features'], label='id')
bird_model = tc.nearest_neighbors.create(bird_data, features=['deep_features'], label='id')

nearest_cat_id = get_images_from_ids(cat_model.query(image_test[0:1]))[0]['id']
cat_data[cat_data['id'] == nearest_cat_id]['image'][0].show()

nearest_dog_id = get_images_from_ids(dog_model.query(image_test[0:1]))[0]['id']
dog_data[dog_data['id'] == nearest_dog_id]['image'][0].show()

cat_neighbors_distance_mean = cat_model.query(image_test[0:1])['distance'].mean()

dog_neighbors_distance_mean = dog_model.query(image_test[0:1])['distance'].mean()

cat_test_data = image_test[image_test['label'] == 'cat']
dog_test_data = image_test[image_test['label'] == 'dog']
car_test_data = image_test[image_test['label'] == 'automobile']
bird_test_data = image_test[image_test['label'] == 'bird']

dog_cat_neighbors = cat_model.query(dog_test_data, k=1)
dog_car_neighbors = car_model.query(dog_test_data, k=1)
dog_bird_neighbors = bird_model.query(dog_test_data, k=1)
dog_dog_neighbors = dog_model.query(dog_test_data, k=1)

distances = tc.SFrame()
distances['dog-dog'] = dog_dog_neighbors['distance']
distances['dog-cat'] = dog_cat_neighbors['distance']
distances['dog-car'] = dog_car_neighbors['distance']
distances['dog-bird'] = dog_bird_neighbors['distance']


def is_dog_correct(row):
    return 1 if (row['dog-dog'] < row['dog-cat']
                 and row['dog-dog'] < row['dog-car']
                 and row['dog-dog'] < row['dog-bird']) else 0


correct_rows = distances.apply(is_dog_correct).sum()

