import ex2_1_2 as ex2
import pandas as pd
import numpy as np
import time


def transform(ratings, min_items=5, min_users=7):
    """
    Transforms ratings DataFrame
    :param ratings: Ratings DataFrame
    :param min_items: Minimum items per user
    :param min_users: Minimum users per item
    :return: Transformed DataFrame
    """
    ratings = ratings.groupby('user').filter(lambda items: len(items) >= min_items)
    ratings = ratings.groupby('item').filter(lambda users: len(users) >= min_users)
    unique_users = ratings['user'].unique()
    unique_items = ratings['item'].unique()
    user_mapping = {u: k for k, u in enumerate(unique_users)}
    item_mapping = {i: k for k, i in enumerate(unique_items)}
    ratings = ratings.replace({'user': user_mapping, 'item': item_mapping})
    return ratings


def train_test_split(ratings, train_ratio=0.8):
    """
    Splits ratings per user
    :param ratings: Ratings DataFrame
    :param train_ratio: Percentage of ratings in the train set
    :return: A tuple of train and test DataFrames
    """
    train, test = [], []
    for user in ratings.groupby('user'):
        rows = user[1].values.tolist()
        split = int(train_ratio * len(rows))
        indices = np.random.permutation(len(rows))
        for i, row in enumerate(rows):
            if i in indices[:split]:
                train.append(row)
            else:
                test.append(row)
    train = pd.DataFrame(train, columns=ratings.columns, index=None)
    test = pd.DataFrame(test, columns=ratings.columns, index=None)
    return train, test


def main():
    ratings = transform(pd.read_csv('ratings.csv'))
    train, test = train_test_split(ratings)
    # PART A
    start = time.time()
    baseline_recommender = ex2.BaselineRecommender(train)
    print(f'BaselineRecommender init took {time.time() - start:.2f}s')
    print(baseline_recommender.rmse(test))
    print(f'BaselineRecommender predict took {time.time() - start:.2f}s')
    #
    neighborhood_recommender = ex2.NeighborhoodRecommender(train)
    print(f'neighborhood_recommender init took {time.time() - start:.2f}s')
    print(neighborhood_recommender.rmse(test))
    print(f'neighborhood_recommender predict (total) took {time.time() - start:.2f}s')
    #
    # PART B
    start = time.time()
    ls_recommender = ex2.LSRecommender(train)
    print(f'ls_recommender init took {time.time() - start:.2f}s')
    ls_recommender.solve_ls()
    print(f'ls_recommender solve_ls took {time.time() - start:.2f}s')
    print(ls_recommender.rmse(test))
    print(f'ls_recommender predict (total) took {time.time() - start:.2f}s')

    # PART C
    start = time.time()
    ratings_comp = pd.read_csv('ratings_comp.csv')
    comp_recommender = ex2.CompetitionRecommender(ratings_comp)
    # print(comp_recommender.rmse(test_comp))
    print(f'Comp init time took {time.time() - start:.2f}s')
    print(comp_recommender.rmse(test))
    comp_recommender = ex2.CompetitionRecommender(train)
    print(f'Comp init time took {time.time() - start:.2f}s')
    print(comp_recommender.rmse(test))

    print(f'Comp total time took {time.time() - start:.2f}s')


if __name__ == '__main__':
    np.random.seed(0)
    main()
