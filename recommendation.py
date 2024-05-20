from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def create_mappings(df):
    """
    Maps user to movie and movie to user

    Args:
        df: pandas dataframe

    Returns:
        user2movie: dictionary that maps user to movie
        movie2user: dictionary that maps movie to user

    """


    user2movie = df.groupby('userId')['movieId'].unique().to_dict()
    movie2user = df.groupby('movieId')['userId'].unique().to_dict()

    return user2movie, movie2user


def create_utility_matrix(df):
    """
    Creates a pivot table from the original DataFrame. This results
    in a new DataFrame where users and movies are organized along rows
    and columns

    Args:
        df: pandas dataframe

    Returns:
        utility_matrix: utility matrix with information about movie ratings

    """
    utility_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
    
    # Fill NaN values with 0
    utility_matrix.fillna(0, inplace=True) 

    return utility_matrix

def create_chronological_mapping(df):
    """
    Creates movie mapping to make it chronological

    Args:
        df: pandas dataframe

    Returns:
        df: pandas dataframe but chronological
    """
    mapping_dict = {}
    for index, value in enumerate(df['movieId'].unique()):
        mapping_dict[value] = index

    df['movieIndex'] = df['movieId'].map(mapping_dict)

def shrinking_data(n, df):
    user_likes = df.groupby('userId')['movieId'].count()
    top_users = user_likes.nlargest(n).index

    top_users_df = df[df['userId'].isin(top_users)].reset_index(drop=True)

    return top_users_df


def movie_cluster(user_id, movie_id, no_of_nearest_neighbors, utility_matrix_train, movies):

    """
    Creates dataframe with recommended movies based on cosine similarity and calculates
    the rating from a given movie based on recommendations for similar movies

    Args:
        user_id: id of user for which we look for recommendation
        movie_id: id of movie for which we look for recommendation 
        no_of_nearest_neighbors: number of clusters
        utility_matrix_train: train matrix created in preprocessing method with movieId, userId and ratingId
        movies: dataet with movies

    Returns:
        df: pandas dataframe with name and id of recommended movies
        estimated_rating: rating we estimated for a given movie based on similar movies

    """

    cf_knn_model= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=no_of_nearest_neighbors, n_jobs=-1)
    cf_knn_model.fit(utility_matrix_train)

    distances, indices = cf_knn_model.kneighbors(utility_matrix_train.T.iloc[:, movie_id].values.reshape(1, -1))
    similar_movies_indices = indices.flatten()[1:]
    similar_movies_ids = utility_matrix_train.columns[similar_movies_indices].tolist()

    cf_recs = []

    for i in similar_movies_ids:
        cf_recs.append({'Movie Id': i, 'Title':movies['title'][i]})

    df = pd.DataFrame(cf_recs, index = range(1, no_of_nearest_neighbors))


    user_ratings = utility_matrix_train.iloc[user_id, similar_movies_ids]

    print("User ratings:", user_ratings)

    weighted_sum = 0
    sum_of_weights = 0

    for rating, distance in zip(user_ratings, distances.flatten()[1:]):
        if not np.isnan(rating) and rating != 0:
            
            weight = 1/(distance + 1e-10)
            weighted_sum += weight * rating
            sum_of_weights += weight
    
    if sum_of_weights != 0:
        estimated_rating = weighted_sum/sum_of_weights
    else:
        estimated_rating = 0

    return df, round(estimated_rating, 2)