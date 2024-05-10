from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


def getTwoDatasets(data):
    """
    Splits data into two datasets: train and test.
    --------
    Input:
    data : pandas.DataFrame
        Input DataFrame.
    --------
    Output:
    pandas.DataFrame, pandas.DataFrame
        Train and Test DataFrames.
    """
    assert isinstance(data, pd.DataFrame), "Input data must be a pandas DataFrame"

    total_rows = len(data)
    split_index = int(0.6 * total_rows)

    train_df = pd.DataFrame(data.iloc[:split_index])
    test_df = pd.DataFrame(data.iloc[split_index:])

    return train_df, test_df

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
