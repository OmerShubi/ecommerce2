import abc
from typing import Tuple
import pandas as pd
import numpy as np
from multiprocessing import  Pool

 # TODO NO GLOBAL PARAMS

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

        true_ratings['prediction'] = true_ratings.apply(lambda x: self.predict(user=int(x[0]), item=int(x[1]), timestamp=x[3]), axis=1)
        rmse = np.sqrt(np.mean((true_ratings['rating'] - true_ratings['prediction'])**2))
        return rmse


class BaselineRecommender(Recommender):
    # TODO runtime 1 minute max
    def initialize_predictor(self, ratings: pd.DataFrame):
        ratings = ratings.copy(deep=True)
        ratings.drop('timestamp', axis=1, inplace=True)
        self.R_hat = ratings.rating.mean()
        self.B_u = ratings.drop('item', axis=1).groupby(by='user').mean().rename(
            columns={'rating': 'user_rating_mean'})

        self.B_u['user_rating_mean'] -= self.R_hat
        self.B_i = ratings.drop('user', axis=1).groupby(by='item').mean().rename(
            columns={'rating': 'item_rating_mean'})
        self.B_i['item_rating_mean'] -= self.R_hat



    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        prediction = self.R_hat + self.B_u.loc[user, 'user_rating_mean'] + self.B_i.loc[item, 'item_rating_mean']
        return float(np.clip(prediction, a_min=0.5, a_max=5))


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        ratings = ratings.copy(deep=True)
        ratings.drop('timestamp', axis=1, inplace=True)
        self.R_hat = ratings.rating.mean()
        self.B_u = ratings.drop('item', axis=1).groupby(by='user').mean().rename(columns={'rating': 'user_rating_mean'})

        self.B_u['user_rating_mean'] -= self.R_hat
        self.B_i = ratings.drop('user', axis=1).groupby(by='item').mean().rename(columns={'rating': 'item_rating_mean'})
        self.B_i['item_rating_mean'] -= self.R_hat

        ratings['rating_adjusted'] = ratings['rating']-self.R_hat
        num_users = len(self.B_u)
        num_items = len(self.B_i)

        R_tilde = np.zeros((num_items, num_users))
        R_tilde[ratings.item.values.astype(int), ratings.user.values.astype(int)] = ratings.rating_adjusted.values

        self.R_tilde = pd.DataFrame(R_tilde).astype(pd.SparseDtype("float", 0.0))

        self.user_corr = self.R_tilde.corr(method=NeighborhoodRecommender.custom_corr)
        self.groupby_per_item = ratings.groupby(by='item')# TODO drop ratings and rating_adjusted

    @staticmethod
    def custom_corr(a, b):
        common_indices = np.intersect1d(np.nonzero(a)[0], np.nonzero(b)[0])
        # if no common ratings
        if common_indices.size == 0:
            return 0
        corr = np.dot(a, b) / (np.linalg.norm(a[common_indices]) * np.linalg.norm(b[common_indices]))
        return corr

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        neighbours = set(self.groupby_per_item.get_group(item)['user'].values) - {user}
        nearest_neighbours_corr = self.user_corr[user][neighbours].sort_values(ascending=False).iloc[:3]
        nearest_neighbours_ratings = self.R_tilde.loc[int(item), nearest_neighbours_corr.index]
        neighbour_deviation = (nearest_neighbours_corr*nearest_neighbours_ratings).sum()/nearest_neighbours_corr.abs().sum()

        prediction = self.R_hat + self.B_u.loc[user, 'user_rating_mean'] + self.B_i.loc[item, 'item_rating_mean'] + neighbour_deviation
        return float(np.clip(prediction, a_min=0.5, a_max=5))

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
