import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules



def shrinking_data(n: int, df: pd.DataFrame) -> pd.DataFrame:


    assert isinstance(n, int), "n should be an integer"
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"
    assert 'userId' in df.columns, "userId column is missing"
    assert 'movieId' in df.columns, "movieId column is missing"

    user_likes = df.groupby('userId')['movieId'].count()
    top_users = user_likes.nlargest(n).index

    top_users_df = df[df['userId'].isin(top_users)].reset_index(drop=True)

    return top_users_df


def return_datasets(data: pd.DataFrame, kf: int) -> tuple:

    """
    Combines datsets after KFold method

    Args:
        data: pandas dataframe which is splitted
        kf: kfold

    Returns:
        combined_train_data: all train subsets combined together
        combined_test_data: all test subsets combined together
    """

    ### TESTS
    assert isinstance(data, pd.DataFrame), "data should be a DataFrame"

    data = data.copy()
    all_train_data = {}
    all_test_data = {}
    index = 0

    for train_index, test_index in kf.split(data):
    
        train_data_kf = data.iloc[train_index].copy()
        test_data_kf = data.iloc[test_index].copy()

        all_train_data[index] = train_data_kf
        all_test_data[index] = test_data_kf
        index += 1


    return all_train_data, all_test_data


def train_kmeans_and_predict(train_df: dict, test_df: dict, n_clusters=5) -> tuple:
    """
    Trains KMeans model and predicts ratings for the test data

    Args:
        train_df: dictonary of train subsets for each KFold split
        test_df: dictonary of test subsets for each KFold split
        n_clusters: number of clusters for KMeans

    Returns:
        mse_list: list of mean squared errors for each KFold split
        models: list of trained KMeans models
        train_dfs: dictonary of train dataframes with clusters assigned
        test_dfs: dictonary of test dataframes with predicted ratings
    """
    ### TESTS
    assert isinstance(train_df, dict), "train_df should be a dictonary"
    assert isinstance(test_df, dict), "test_df should be a dictonary"
    assert isinstance(n_clusters, int), "n_clusters should be an integer"

    mse_list = []
    models = []
    train_dfs = {}
    test_dfs = {}
    index = 0

    for key in train_df:
        train_data = train_df[key]
        test_data = test_df[key]

        ### TESTS
        assert isinstance(train_data, pd.DataFrame), "each element in train_df should be a DataFrame"
        assert isinstance(test_data, pd.DataFrame), "each element in test_df should be a DataFrame"
        assert 'userId' in train_data.columns, "userId column is missing in train_df"
        assert 'movieId' in train_data.columns, "movieId column is missing in train_df"
        assert 'rating' in train_data.columns, "rating column is missing in train_df"
        assert 'title' in train_data.columns, "title column is missing in train_df"

        X_train = train_data.drop(columns=['userId', 'movieId', 'rating', 'title'])
        X_test = test_data.drop(columns=['userId', 'movieId', 'rating', 'title'])
        y_test = test_data['rating']

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)

        train_data['cluster'] = kmeans.labels_
        
        if 'predicted_rating' in test_data.columns:
            test_data = test_data.drop(columns=['predicted_rating'])
        
        test_data['cluster'] = kmeans.predict(X_test)

        cluster_mean_ratings = train_data.groupby('cluster')['rating'].mean()
        test_data['predicted_rating'] = test_data['cluster'].map(cluster_mean_ratings)

        mse = mean_absolute_error(y_test, test_data['predicted_rating'])
        mse_list.append(mse)

        models.append(kmeans)
        train_dfs[index] = train_data
        test_dfs[index] = test_data
        index += 1

    return mse_list, models, train_dfs, test_dfs

def predict_rating_kmeans(user_id:int, movie_id:int, models, train_dfs: dict, movie_data: pd.DataFrame):
    """
    Predicts the rating for a given user and movie

    Args:
        user_id: ID of the user
        movie_id: ID of the movie
        models: list of trained KMeans models
        train_dfs: dictonary of train dataframes with clusters assigned
        movie_data: original movie data to find movie features

    Returns:
        predicted_rating: predicted rating for the given user and movie
    """
    ### TESTS
    assert isinstance(user_id, int), "user_id should be an integer"
    assert isinstance(movie_id, int), "movie_id should be an integer"
    assert isinstance(models, list), "models should be a list"
    assert isinstance(train_dfs, dict), "train_dfs should be a dictonary"
    assert isinstance(movie_data, pd.DataFrame), "movie_data should be a DataFrame"
    assert 'userId' in movie_data.columns, "userId column is missing in movie_data"
    assert 'movieId' in movie_data.columns, "movieId column is missing in movie_data"
    assert 'rating' in movie_data.columns, "rating column is missing in movie_data"
    assert 'title' in movie_data.columns, "title column is missing in movie_data"
    
    # Find the movie features
    movie_features = movie_data[movie_data['movieId'] == movie_id].drop(columns=['userId', 'movieId', 'rating', 'title'])
    if movie_features.empty:
        return None  # Movie not found

    # Predict cluster for the movie using the first model (for simplicity)
    predicted_cluster = models[0].predict(movie_features)

    # Find the mean rating for the predicted cluster in the train data
    cluster_mean_ratings = train_dfs[0].groupby('cluster')['rating'].mean()
    predicted_rating = cluster_mean_ratings.get(predicted_cluster[0], None)

    return predicted_rating, predicted_cluster, cluster_mean_ratings

def getFrequentItemset(clustered_data: dict, min_support: float=0.1) -> dict:
    """
    Finds frequent itemsets from the clustered data

    ------------
    Args:
        clustered_data: dictonary of train dataframes with clusters assigned
        min_support: minimum support for the frequent itemsets

    ------------
    Returns:
        frequent_itemsets: dictonary of frequent itemsets found for each train dataframe for each cluster
    """

    ### TESTS
    assert isinstance(clustered_data, dict), "clustered_data should be a dictonary"
    assert isinstance(min_support, float), "min_support should be a float"
    assert min_support > 0 and min_support < 1, "min_support should be between 0 and 1"

    frequent_itemsets = {}
    index = 0

    for key in clustered_data:

        dataset = clustered_data[key]
        
        ### TESTS
        assert isinstance(dataset, pd.DataFrame), "each element in clustered_data should be a DataFrame"
        assert 'cluster' in dataset.columns, "cluster column is missing"

        curr_freq_itemset = []
        train_df = dataset.drop(columns=['userId', 'movieId', 'rating', 'title'])

        for cluster in train_df['cluster'].unique():
            freq = apriori(train_df[train_df['cluster']==cluster].drop(columns=['cluster']), min_support, use_colnames=True)
            curr_freq_itemset.append(freq)

        frequent_itemsets[index] = curr_freq_itemset
        index += 1
    
    return frequent_itemsets


def getRules(freq_itemsets: dict, metric: str='confidence', min_threshold: float=0.8) -> dict:
    """
    Generates association rules from the frequent itemsets

    ------------
    Args:
        freq_itemsets: dictonary of frequent itemsets for each cluster
        metric: metric used to assess the quality of association rules
        min_threshold: minimum threshold which the metric has to fulfill

    ------------
    Returns:
        rules: dictonary of association rules for each cluster in each group of freq_itemsets
    """

    ### TESTS
    assert isinstance(freq_itemsets, dict), "freq_itemsets should be a dictonary"
    assert isinstance(metric, str), "metric should be a string"
    assert metric in ['confidence', 'lift', 'leverage', 'conviction'], "metric should be one of 'confidence', 'lift', 'leverage', 'conviction"
    assert isinstance(min_threshold, float), "min_threshold should be a float"
    assert min_threshold > 0 and min_threshold < 1, "min_threshold should be between 0 and 1"

    rules = {}
    index = 0

    for key in freq_itemsets:

        set = freq_itemsets[key]
        ### TESTS
        assert isinstance(set, list), "each element in freq_itemsets[] should be a list"

        curr_rules = []
        for cluster in set:

            ### TESTS
            assert isinstance(cluster, pd.DataFrame), "each element in freq_itemsets[][] should be a DataFrame"

            assoc_rules = association_rules(cluster, metric, min_threshold)
            curr_rules.append(assoc_rules)

        rules[index] = curr_rules
        index += 1
    
    return rules

def filterRules(userId: int, movieId: int, df: pd.DataFrame, cluster: int, cluster_mean_ratings: pd.Series, rules: dict) -> dict:
    """
    Filters rules to contain only rules important for a given user and movie based on the rules generated by the Apriori algorithm.

    ------------
    Args:
        userId: ID of the user
        movieId: ID of the movie
        df: DataFrame containing the user-movie data
        cluster: Cluster predicted for the movie
        cluster_mean_ratings: 
        rules: Rules generated by the Apriori algorithm

    ------------
    Returns:
        important_rules: Filtered rules or 
                1) 'rated', rating, mess if the user has already rated the movie
                2) 'no rated', cluster_mean_ratings[cluster], mess if the user has not rated any movies yet
                3) 'no rules', cluster_mean_ratings[cluster], mess if there are no rules for the movie

    """

    ### TESTS
    assert isinstance(userId, int), "userId should be an integer"
    assert isinstance(movieId, int), "movieId should be an integer"
    assert isinstance(df, pd.DataFrame), "df should be a DataFrame"
    assert 'userId' in df.columns, "userId column is missing"
    assert 'movieId' in df.columns, "movieId column is missing"
    assert 'rating' in df.columns, "rating column is missing"
    assert isinstance(cluster, int), "cluster should be an integer"
    assert isinstance(cluster_mean_ratings, pd.Series), "cluster_mean_ratings should be a pandas Series"
    assert isinstance(rules, dict), "rules should be a dictonary"


    user_movies = df[df['userId'] == userId]
    
    if movieId in user_movies['movieId'].values:
        rating = user_movies[user_movies['movieId'] == movieId]['rating'].values[0]
        mess = f"User {userId} has already rated movie {movieId}"
        return {'code': 'rated', 'rating': rating, 'mess': mess}
    
    elif user_movies.empty:
            mess = f"User {userId} has not rated any movies yet"
            return {'code': 'no rated', 
                'rating': cluster_mean_ratings[cluster], 
                'mess': mess}
    
    else:
        user_movies = user_movies.drop(columns = ['userId', 'movieId', 'title'])
        user_movies['genres'] = user_movies.apply(lambda x: list(x.index[x == 1]), axis=1)
        user_movies = user_movies[['rating', 'genres']]
        
    movie_genres = df[df['movieId'] == movieId].drop(columns=['userId', 'movieId', 'rating', 'title']).where(lambda x: x == 1).dropna(axis=1).columns
    
    important_rules = {}
    index = 0

    for key in rules:
        curr_import_rules = []

        for rule in rules[key][cluster].iterrows():

            ### TESTS
            assert isinstance(rule[1], pd.Series), "each element in rules[] should be a pandas Series"
            assert 'consequents' in rule[1].index, "consequents column is missing"

            if rule[1]['consequents'].issubset(movie_genres):
                for movie in user_movies.iterrows():

                    ### TESTS
                    assert 'antecedents' in rule[1].index, "antecedents column is missing"

                    if rule[1]['antecedents'].issubset(movie[1][1]):
                        curr_import_rules.append((rule[1], movie[1][0]))

        if curr_import_rules:
            important_rules[index] = curr_import_rules
            index += 1

    if not important_rules:
        mess = f"No rules found for movie {movieId}"
        return {'code': 'no rules', 
                'rating': cluster_mean_ratings[cluster], 
                'mess': mess}
    
    return important_rules

def predictRatingRules(important_rules: dict) -> float:
    """
    Predicts the rating for a movie based on the rules generated by the Apriori algorithm.

    ------------
    Args:
        important_rules: Important rules generated by the filterRules function

    ------------
    Returns:
        predicted_rating: Predicted rating for the movie

    """

    ### TESTS
    assert isinstance(important_rules, dict), "important_rules should be a dictonary"

    if 'code' in important_rules.keys():

        ### TESTS
        assert isinstance(important_rules['rating'], float), "second element in important_rules should be a float if the first element is 'rated', 'no rated' or 'no rules'"
        # print(important_rules['mess'])
        return important_rules['rating']
    
    predicted_rating = 0
    models = 0

    model_rating = 0
    model_confidence = 0

    for key in important_rules:
        for rule in important_rules[key]:

            ### TESTS
            assert isinstance(rule, tuple), "each element in important_rules should be a tuple"
            assert isinstance(rule[0], pd.Series), "first element in the tuple should be a pandas Series"
            assert 'confidence' in rule[0].index, "confidence column is missing"
            assert isinstance(rule[1], float), "second element in the tuple should be a float"

            
            model_rating += rule[0]['confidence'] * rule[1]
            model_confidence += rule[0]['confidence']

        if model_confidence != 0:
            predicted_rating += model_rating/model_confidence
            models += 1

    if models == 0:
        return 0
    
    rating = round(predicted_rating/models, 2)
    return rating

def getErrorRules(testdf: dict, rules: dict, cluster_mean_ratings: pd.Series) -> float:
    """
    Calculates the mean absolute error for the rules-based recommendation system

    ------------
    Args:
        testdf: dict of DataFrames containing the test data
        rules: Rules generated by the Apriori algorithm
        cluster_mean_ratings: Mean ratings for each cluster

    ------------
    Returns:
        error: Mean absolute error for the rules-based recommendation system for each model
    """

    ### TESTS
    assert isinstance(testdf, dict), "testdf should be a dictonary"
    assert isinstance(rules, dict), "rules should be a dictonary"
    assert isinstance(cluster_mean_ratings, pd.Series), "cluster_mean_ratings should be a pandas Series"

    error = 0

    schuffle = testdf[0].sample(frac=1)
    test_data = schuffle[:3000]

    error_curr = 0
    
    ### TESTS
    assert isinstance(test_data, pd.DataFrame), "each element in testdf should be a DataFrame"
    assert 'userId' in test_data.columns, "userId column is missing"
    assert 'movieId' in test_data.columns, "movieId column is missing"
    assert 'rating' in test_data.columns, "rating column is missing"
    assert 'cluster' in test_data.columns, "cluster column is missing"

    for index, row in test_data.iterrows():
        
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        cluster = row['cluster']
        #remove the current row from the dataset
        filtered_rules = filterRules(user_id, movie_id, test_data.drop(index), cluster, cluster_mean_ratings, rules)
        predicted_rating = predictRatingRules(filtered_rules)

        
        error_curr += abs(rating - predicted_rating)
    
    error += error_curr/len(test_data)

    return error/len(testdf)


def roundRating(rating: float) -> float:
    """
    Rounds the rating to the nearest half integer
    
    ------------
    Args:
        rating: rating to be rounded to 2 decimal places
    
    ------------
    Returns:
        rounded_rating: rounded rating
    """

    ### TESTS
    assert isinstance(rating, float), "rating should be a float"

    decimal = rating - int(rating)
    if decimal < 0.25:
        return int(rating)
    elif decimal < 0.75:
        return int(rating) + 0.5
    else:
        return int(rating) + 1

