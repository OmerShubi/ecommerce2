import abc
from typing import Tuple
import pandas as pd
import numpy as np

# TODO rename filename

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
        pass


class BaselineRecommender(Recommender):
    # TODO runtime 1 minute max
    def initialize_predictor(self, ratings: pd.DataFrame):
        ratings.drop('timestamp', axis=1, inplace=True)
        self.R_hat = ratings.rating.mean()
        self.B_u = ratings.drop('item', axis=1).groupby(by='user').mean()
        self.B_i = ratings.drop('user', axis=1).groupby(by='item').mean()
        pass


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.R_hat + self.B_u.loc[user] + self.B_i.loc[item]
        return float(np.clip(prediction, a_min=0.5, a_max=5))


class NeighborhoodRecommender(Recommender):
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

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        pass


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
