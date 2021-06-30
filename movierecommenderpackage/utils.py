from fuzzywuzzy import process
import pandas as pd
import numpy as np
import pickle
import os

package_dir = os.path.dirname(__file__)

# put the movieId into the row index!
movies = pd.read_csv(package_dir + '/data/ml-latest-small/movies.csv', index_col=0) 

# one hot encoded genres (from movie['genres'])
movie_genres = pd.read_csv(package_dir + '/data/ml-latest-small/movies_genres.csv', index_col=0)

# combining movies and genres
movies_genres_ohe = pd.concat([movies, movie_genres.loc[:,'romance':]], axis=1)

genres = ['romance', 'sci-fi', 'animation', 'film-noir', 'musical', 'adventure', 'thriller',
        'horror', 'documentary', 'fantasy', 'mystery', 'children', 'comedy',
        'crime', 'western', 'imax', 'war', 'drama', 'action']


#import ratings and transform
ratings_long = pd.read_csv(package_dir + '/data/ml-latest-small/ratings.csv')
ratings = pd.pivot_table(ratings_long, 
                        values='rating', 
                        index='userId', 
                        columns='movieId')
movie_average_rating = ratings.mean(axis=0)

dummy_user = {
    'ratings': {'Titanic': '3', 'Iron Man':'4', 'Sherlock Holmes': '4', 'Lion King': '5'},
    'genres': ['action', 'crime', 'children']
}


def lookup_movie(search_query, titles):
    """
    given a search query, uses fuzzy string matching to search for similar 
    strings in a pandas series of movie titles

    returns a list of search results. Each result is a tuple that contains 
    the title, the matching score and the movieId.
    """
    matches = process.extractBests(search_query, titles)
    # [(title, score, movieId), ...]
    return matches

def watched_movies(userId):
    '''
    this function creates a user-item-matrix of the ratings_long table
    and returns a list of all watched movies by a user (userId)
    '''
    ratings = pd.pivot_table(ratings_long, values='rating', index='userId', columns='movieId')
    watched_movies = ratings.loc[userId].dropna().index
    return list(watched_movies)

def load_model(path):
    '''
    returns a pre-trained model
    '''
    model = pickle.load(open(path, 'rb'))
    return model

def replace_titles_with_ids(user, titles):
    '''
    returns a dictionary with movie ids instead of titles
    '''
    new_user = {
            'ratings':{}
    }
    new_user['genres']=user['genres']
    for key in user['ratings'].keys():
        movie_id = lookup_movie(key, titles)[0][2]
        new_user['ratings'][movie_id] = user['ratings'][key]
    return new_user
    

def create_new_user(user, rec_type='neighbor'):
    '''
    returns a vector which includes the new user ratings
    the vector has different shapes, based on different recommenders
    neighbor: vector for a sparse-matrix with 193610 columns, initialized with 0 for unrated movies
    NMF: vector with length of ratings columns, initialized with 0 for unrated movies 
    '''
    new_user = replace_titles_with_ids(user, movies['title'])
    if rec_type == 'neighbor':
        vector = np.repeat(0, 193610)
        for key,val in new_user['ratings'].items():
            vector[key] = int(val)
    elif rec_type =='NMF':
        vector = np.repeat(0, len(ratings.columns))
        for key,val in new_user['ratings'].items():
            vector[ratings.columns.tolist().index(key)] = int(val)
    return vector

def create_neighborhood(user_vector):
    '''
    applies a pre-trained nearest neighbor model to the data
    returns an array with user ids
    '''
    model = load_model(package_dir + '/models/NN_cosine.sav')
    distances, neighbor_ids = model.kneighbors(user_vector, n_neighbors=20)
    return neighbor_ids

def create_genre_index_filter(user):
    '''
    returns a pandas index of movie ids, filteres by favorite genre
    '''
    df = pd.DataFrame(columns=movies_genres_ohe.columns)
    for genre in user['genres']:
        df = pd.concat([df, movies_genres_ohe[movies_genres_ohe[genre] ==1]])
    df.drop_duplicates(inplace=True)
    return df.index




if __name__ == '__main__':
    results = lookup_movie('star wars', movies['title'])
    print(results)

