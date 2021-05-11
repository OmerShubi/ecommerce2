import abc
from typing import Tuple
import pandas as pd
import numpy as np
from multiprocessing import  Pool
# from sklearn.metrics import mean_squared_error todo
# TODO rename filename


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df





class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """

        ratings = pd.merge(true_ratings, self.B_u, on='user', how='left', sort=False)
        ratings = pd.merge(ratings, self.B_i, on='item', how='left', sort=False)

        ratings['base_prediction'] = self.R_hat + ratings['user_rating_mean'] + ratings['item_rating_mean']
        ratings['base_prediction'] = np.clip(ratings['base_prediction'], a_min=0.5, a_max=5)

        # Ratings['rating_adjusted'] = Ratings['rating']-Ratings['rating_mean']
        # res = parallelize_dataframe(true_ratings, self.rmse_split, n_cores=4)

        rmse = np.sqrt(np.mean((ratings['rating'] - ratings['base_prediction'])**2))
        return rmse
    #
    # def rmse_split(self, true_ratings):
    #     true_ratings['prediction'] = true_ratings.apply(lambda x: self.predict(user=x[0],  item=x[1], timestamp=x[3]),
    #                                                     axis=1)
    #     return true_ratings

class BaselineRecommender(Recommender):
    # TODO runtime 1 minute max
    def initialize_predictor(self, ratings: pd.DataFrame):
        ratings = ratings.copy(deep=True)
        ratings.drop('timestamp', axis=1, inplace=True)
        self.R_hat = ratings.rating.mean()
        self.B_u = ratings.drop('item', axis=1).groupby(by='user', as_index=False, sort=False).mean().rename(
            columns={'rating': 'user_rating_mean'})
        self.B_i = ratings.drop('user', axis=1).groupby(by='item', as_index=False, sort=False).mean().rename(
            columns={'rating': 'item_rating_mean'})


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.R_hat + self.B_u.loc[self.B_u.user == user, 'user_rating_mean'].values + self.B_i.loc[self.B_i.item == item,'item_rating_mean'].values
        return float(np.clip(prediction, a_min=0.5, a_max=5))


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        ratings = ratings.copy(deep=True)
        ratings.drop('timestamp', axis=1, inplace=True)
        self.R_hat = ratings.rating.mean()
        self.B_u = ratings.drop('item', axis=1).groupby(by='user', as_index=False, sort=False).mean().rename(
            columns={'rating': 'user_rating_mean'})
        self.B_i = ratings.drop('user', axis=1).groupby(by='item', as_index=False, sort=False).mean().rename(
            columns={'rating': 'item_rating_mean'})

        ratings['rating_adjusted'] = ratings['rating']-self.R_hat
        num_users =len(self.B_u)
        num_items =len(self.B_i)

        R_tilde = np.zeros((num_items, num_users))
        R_tilde[ratings.item.values.astype(int), ratings.user.values.astype(int)] = ratings.rating_adjusted.values

        df = pd.DataFrame(R_tilde).astype(pd.SparseDtype("float", 0.0))
        self.user_corr = df.corr()

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        corr = self.user_corr.loc[user1, user2]
        return corr


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        pass


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass
