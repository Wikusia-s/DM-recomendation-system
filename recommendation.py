from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


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


def shrinking_data(n, df):
    user_likes = df.groupby('userId')['movieId'].count()
    top_users = user_likes.nlargest(n).index

    top_users_df = df[df['userId'].isin(top_users)].reset_index(drop=True)

    return top_users_df


def return_datasets(data, kf):

    """
    Combines datsets after KFold method

    Args:
        data: pandas dataframe which is splitted
        kf: kfold

    Returns:
        combined_train_data: all train subsets combined together
        combined_test_data: all test subsets combined together
    """

    data = data.copy()
    all_train_data = []
    all_test_data = []
    
    for train_index, test_index in kf.split(data):
    
        train_data_kf = data.iloc[train_index].copy()
        test_data_kf = data.iloc[test_index].copy()

        all_train_data.append(train_data_kf)
        all_test_data.append(test_data_kf)


    return all_train_data, all_test_data


