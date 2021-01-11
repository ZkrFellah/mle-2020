import pandas as pd
import numpy as np

from .helpers import  get_movie_id, get_movie_name, get_movie_year

class BaseRecommander:
    """Base Model
    Inherit from this class to ensure reusability and compatibility of new model    
    """
    def fit(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError

class ContentFilter(BaseRecommander):
    
    def fit(self, movie_df, ratings, genre_cols=None):
        """fit the content filter model

        Args:
            movie_df (pd.DataFrame): Dataframe of movies contains cols: movie_id | title | year | *genres
            ratings (pd.DataFrame): DataFrame contains ratings cols: user_id | movie_id | rating
            genre_cols (List): list of movie genres. Defaults to None

        """        
        if genre_cols is None: # if genre_cols are not specified calculate them from movies dataframe
            genre_cols = movie_df.columns.difference(['movie_id', 'title', 'year']).values
        self.similarity = movie_df[genre_cols].values.dot(movie_df[genre_cols].values.T)
        self.ratings = ratings
        self.movies = movie_df
        
    def _get_most_similar(self, movie_name, year=None, top=10):
        """ find the most similar movies to ref movie

        Args:
            movie_name (int): movie_name
            year (int, optional): year. Defaults to None.
            top (int, optional): number of similar movies. Defaults to 10.

        Returns:
            List: list of tuple(id_movie, name, similarity)
        """        
        index_movie = get_movie_id(self.movies, movie_name, year)
        best = self.similarity[index_movie].argsort()[::-1]
        return [(ind, get_movie_name(self.movies, ind), self.similarity[index_movie, ind]) for ind in best[:top] if ind != index_movie]
    
    def predict_df(self, user_id, N=5, M=10):
        """suggest top N movies in a pd.DataFrame Format cols: movie_id | title | similarity

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.
            M (int, optional): number of rated movies used to find similar unwatched movies. Defaults to 10.

        Returns:
            pd.DataFrame: columns : movie_id | title | similarity
        """        
    
        top_movies = self.ratings[self.ratings['user_id'] == user_id].sort_values(by='rating', ascending=False).head(M)['movie_id']
        index=['movie_id', 'title', 'similarity']
        most_similars = []
        for top_movie in top_movies:
            most_similars += self._get_most_similar( get_movie_name(self.movies, top_movie), get_movie_year(self.movies, top_movie))
        return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity', ascending=False).head(5)
    
    def predict(self, user_id, N=5, M=10):
        """predict the ids of top N movies

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.
            M (int, optional): number of rated movies used to find similar unwatched movies. Defaults to 10.

        Returns:
            np.Array: of movie ids
        """        
        prediction = self.predict_df(user_id, N=N, M=M)
        return prediction['movie_id'].values
    
    def predict_sim(self, user_id, N=5, M=10):
        """predict the ids of top N movies with their similarities

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.
            M (int, optional): number of rated movies used to find similar unwatched movies. Defaults to 10.

        Returns:
            pd.Series: of movie ids as index and similarity as values
        """   
        prediction = self.predict_df(user_id, N=N, M=M)
        return prediction[['movie_id', 'similarity']].set_index('movie_id')


class SVD(BaseRecommander):
    
    def fit(self, ratings, k=20):
        """fit the collaborative filter model

        Args:
            ratings (pd.DataFrame): DataFrame contains ratings cols: user_id | movie_id | rating
            k (int, optional): number of component to reconstruct the matrix. Defaults to 20.
        """ 

        self.df_ratings = ratings.pivot(
            index='user_id',
            columns='movie_id',
            values='rating').fillna(0)

        U, S, Vt = np.linalg.svd(self.df_ratings.values, full_matrices=False)
        
        predicted_R = (U[:,0] * S[0]).reshape([len(U),1]).dot(Vt[0,:].reshape([1, len(Vt)])) 
        for i in np.arange(1,k):
            predicted_R = predicted_R + (U[:,i] * S[i]).reshape([len(U),1]).dot(Vt[i,:].reshape([1, len(Vt)])) 

        self.predicted_R_df = pd.DataFrame(predicted_R, index = self.df_ratings.index, columns = self.df_ratings.columns)

    def predict_ratings(self, user_id, N=10):
        """predict the ids of top N movies with their predicted ratings

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.

        Returns:
            pd.Series: of movie ids as index and predicted ratings as values
        """   
        user_movies = self.df_ratings.loc[user_id]
        predicted_rated_movies = self.predicted_R_df.loc[user_id]
        recommended_movies = predicted_rated_movies[user_movies==0].sort_values(ascending=False)[0:N]
        return recommended_movies
    def predict(self, user_id, N=5):
        """predict the ids of top N movies

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.

        Returns:
            np.Array: of movie ids
        """      
        return np.array(self.predict_ratings(user_id, N).index)

class CollaborativeFilter(BaseRecommander):
    
    def fit(self, ratings):
        """fit the collaborative filter model

        Args:
            ratings (pd.DataFrame): DataFrame contains ratings cols: user_id | movie_id | rating
        """ 

        self.df_movie_features = ratings.pivot(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
    def predict_sim(self, user_id, N=5):
        """predict the ids of top N movies with their similarities

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.

        Returns:
            pd.Series: of movie ids as index and similarity as values
        """   
        user_ratings = self.df_movie_features.loc[user_id]
        similair_users = self.df_movie_features.T.corrwith(user_ratings)
        not_ratated_movies = user_ratings[user_ratings == 0].index
        predicted = similair_users.dot(self.df_movie_features.loc[:, not_ratated_movies]) / sum(similair_users)
        return predicted.sort_values(ascending=False)[:N]
    
    def predict(self, user_id, N=5):
        """predict the ids of top N movies

        Args:
            user_id ([int]): id for user
            N (int, optional): number of movies to suggest. Defaults to 5.

        Returns:
            np.Array: of movie ids
        """      
        return np.array(self.predict_sim(user_id, N).index)

    
