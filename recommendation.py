from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def create_utility_matrix(df):
    """
    Generates a sparse matrix from ratings fataframe.

    Args:
        df: pandas dataframe containing 3 columns (userId, movieId, rating)

    Returns:
        utility_matrix: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that map movie indices to movie id's

    """

    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df['userId']), list(range(M))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df['userId'])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df['movieId'])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    utility_matrix = csr_matrix((df['rating'], (user_index, item_index)), shape = (M,N))

    return utility_matrix, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


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