# song recommender

import turicreate as tc
import matplotlib.pyplot as plt

# load music data
song_data = tc.SFrame.read_csv('../data/song_data.csv')

# how many users are involved?
# count number of users
users = song_data['user_id'].unique()

train_data, test_data = song_data.random_split(.8, seed=0)

# simple popularity-based recommender
popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='song')

# use the popularity model to make some predictions
popularity_model.recommend(users=[users[0]])
popularity_model.recommend(users=[users[1]])
# everyone gets recommended the same things, based on popularity

# create a song recommender
personalized_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='song')

# applying the personalized model to make song recommendations
personalized_model.recommend(users=[users[0]])
personalized_model.recommend(users=[users[1]])

personalized_model.get_similar_items(['With Or Without You - U2'])

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])

# quantitative comparison between the models
model_performance = tc.recommender.util.compare_models(test_data, [popularity_model, personalized_model],
                                                       user_sample=0.15)

plt.scatter(model_performance[0]['precision_recall_overall']['recall'],
            model_performance[0]['precision_recall_overall']['precision'],
            label='popularity model')

plt.scatter(model_performance[1]['precision_recall_overall']['recall'],
            model_performance[1]['precision_recall_overall']['precision'],
            label='popularity model')

plt.xlim([0.0, .13])
plt.ylim([0.0, .05])
plt.xlabel('recall')
plt.ylabel('precision')

plt.show()

# recommending songs assignment

# number of unique users who have listened to songs by various artists
artists = ['Kanye West', 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']

for artist in artists:
    print('{} : {}'.format(artist, len(song_data[song_data['artist'] == artist])))

# using groupby-aggregate to find the mos popular and least popular artist
groupby_result = song_data.groupby(key_column_names='artist',
                                   operations={'total_count': tc.aggregate.SUM('listen_count')})\
    .sort('total_count', ascending=False)

# using groupby-aggregate to find the most recommended songs
subset_test_users = test_data['user_id'].unique()[0:10000]

recommended_songs = personalized_model.recommend(subset_test_users, k=1)
groupby_recommended_songs = recommended_songs.groupby(key_column_names='song',
                                                      operations={'count': tc.aggregate.COUNT()})\
    .sort('count', ascending=False)

