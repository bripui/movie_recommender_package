"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
import os
from movierecommenderpackage.utils import movies, ratings, ratings_long
from movierecommenderpackage.utils import replace_titles_with_ids, create_new_user, create_neighborhood 
from movierecommenderpackage.utils import create_genre_index_filter, load_model

package_dir = os.path.dirname(__file__)

def recommend_random(liked_items, k=5):
    """
    return k random unseen movies for user 
    """
    # dummy implementation
    return movies.sample(k)



def recommend_most_popular(liked_items, movie_item_avg, k=5):
    """
    return k most popular unseen movies for user
    """
    return None

def neighbor_recommender(user):
    """
    return 10 movie title recommendations based on NearestNeighbors algorithm
    """
    #create a new user vector
    user_with_ids = replace_titles_with_ids(user, movies['title'])
    user_vector = create_new_user(user, rec_type='neighbor')
    #find simular users
    neighbor_ids = create_neighborhood([user_vector])
    neighborhood = ratings_long.set_index('userId').loc[neighbor_ids[0]]
    #create recommendations
    recommendations = neighborhood.groupby('movieId')['rating'].sum().sort_values(ascending=False)
    #remove already watched movies
    item_filter = (~recommendations.index.isin(list(user_with_ids['ratings'].keys())))
    genre_index = create_genre_index_filter(user)
    genre_filter = (recommendations.index.isin(genre_index))
    recommendations = recommendations.loc[item_filter&genre_filter]
    return list(movies.loc[recommendations.index]['title'].head(10))


def nmf_recommender(user):
    """
    return 10 movie title recommendations based on non-negative matrix factorization
    """
    user_vector = create_new_user(user, rec_type='NMF')
    user_df = pd.DataFrame(list(user_vector), index=ratings.columns)
    user_df = user_df.T
    model = load_model(package_dir + '/models/NMF_60.sav')
    Q = model.components_
    P = model.transform(user_df)
    prediciton = np.dot(P,Q)
    recommendations = pd.DataFrame(prediciton, columns=ratings.columns)
    final_recs = recommendations[(user_df == 0)].T
    final_recs.columns = ['predicted_rating']
    final_recs= final_recs['predicted_rating'].sort_values(ascending=False)
    return list(movies.loc[final_recs.index]['title'].head(10))


if __name__ == '__main__':
    print(recommend_random([1, 2, 3]))




