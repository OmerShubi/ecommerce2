import abc
from typing import Tuple
import pandas as pd
import numpy as np
from multiprocessing import  Pool
import datetime
 # TODO NO GLOBAL PARAMS
from scipy.sparse.linalg import svds


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


        """
            Calculate Correlation by:
            R^T@R / (R**2@S)^T*(R**2@S)
        """
        self.binary_R_tilde = self.R_tilde != 0
        R_tilde_square = self.R_tilde ** 2
        a = R_tilde_square.transpose().dot(self.binary_R_tilde)
        denominator = a.transpose() * a
        nominator = self.R_tilde.transpose().dot(self.R_tilde)
        corr = nominator / np.sqrt(denominator)
        self.user_corr = corr.fillna(0)
        self.binary_R_tilde = self.binary_R_tilde.sparse.to_dense()
        self.R_tilde = self.R_tilde.sparse.to_dense()

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        nearest_neighbours = self.binary_R_tilde.loc[item]*self.user_corr.loc[user]
        nearest_neighbours.loc[nearest_neighbours == 0] = -1 * float('inf')
        nearest_neighbours_corr = nearest_neighbours.nlargest(n=3)

        nearest_neighbours_ratings = self.R_tilde.loc[int(item), nearest_neighbours_corr.index]
        nominator = (nearest_neighbours_corr*nearest_neighbours_ratings).sum()
        denominator = nearest_neighbours_corr.abs().sum()

        # Handle case where there are no neighbours with correlation
        if nominator == 0 and denominator == 0:
            neighbour_deviation = 0
        else:
            neighbour_deviation = nominator / denominator

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
        ratings = ratings.copy(deep=True)
        ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['weekday'] = pd.to_datetime(ratings['date']).dt.dayofweek  # monday = 0, sunday = 6
        ratings['is_weekend'] = 0
        ratings.loc[ratings['weekday'].isin([4, 5]), 'is_weekend'] = 1
        ratings['is_daytime'] = ratings['date'].dt.time.between(datetime.time(6, 00), datetime.time(18, 00))
        ratings['is_nighttime'] = ~ratings['is_daytime']

        self.R_hat = ratings.rating.mean()

        self.y = ratings['rating'] - self.R_hat
        ratings.drop(['timestamp', 'rating', 'date','weekday'], axis=1, inplace=True)
        ratings = ratings.astype(int)
        self.X = pd.get_dummies(ratings, columns=['user', 'item'], sparse=True)
        self.is_weekend_index = self.X.columns.get_loc('is_weekend')
        self.is_daytime_index = self.X.columns.get_loc('is_daytime')
        self.is_nighttime_index = self.X.columns.get_loc('is_nighttime')

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        # According to result of pd.get_dummies!
        user_index = self.X.columns.get_loc(f'user_{user}')
        item_index = self.X.columns.get_loc(f'item_{item}')
        indices = [user_index, item_index]

        date = datetime.datetime.fromtimestamp(timestamp)

        if date.weekday() in [4, 5]:
            indices.append(self.is_weekend_index)

        if date.time() > datetime.time(6, 00) and date.time() < datetime.time(18, 00):
            indices.append(self.is_daytime_index)
        else:
            indices.append(self.is_nighttime_index)

        prediction = self.R_hat + self.beta[indices].sum()

        return float(np.clip(prediction, a_min=0.5, a_max=5))

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.beta, _, _, _ = np.linalg.lstsq(self.X, self.y, rcond=None)
        return (self.X, self.beta, self.y)


class CompetitionRecommender(Recommender):
    # TODO
    #  Scale instead of clip at 5 and 0.5
    #  More time related features (weekday, quarter, year..)
    #  Abs correlation? K neighbours? Distance measures
    # Item based similarity

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

        u, s, vt = svds(self.R_tilde.transpose(), k=5)
        s_diag = np.diag(s)
        self.X_pred = np.dot(np.dot(u, s_diag), vt)




    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.X_pred[user, item] + self.R_hat

        return float(np.clip(prediction, a_min=0.5, a_max=5))


