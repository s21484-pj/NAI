import argparse
import json
import numpy as np

"""
Authors - Maciej Leciejewski s21484 & Krzysztof Szymczyk s23210

System requirements:
- Python 3.9
- numpy

How to run:
- enter '--user "name last_name"' in Run/Debug Configuration parameters
- execute main function
"""


def build_arg_parser():
    """
    Enables to input user

    :return: argument parser
    """
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True, help='User')

    return parser


def euclidean_score(dataset, user1, user2):
    """
    Compute the Euclidean distance score between user1 and user2

    :param: dataset: contains users, movies and ratings
    :param: user1: name of user1
    :param: user2: name of user2

    :return: the Euclidean distance score between user1 and user2
    """
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    if len(common_movies) == 0:
        return 0
    squared_diff = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def get_users_list(dataset):
    """
    Get users list

    :param: dataset: contains users, movies and ratings

    :return: users list
    """
    user_list = []
    for i in dataset:
        if i != user1:
            user_list.append(i)

    return user_list


def get_matching_results(data, user1, users_list):
    """
    Compute Euclidean and Pearson scores for every user

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: users_list: names of all users

    :return: Euclidean and Pearson scores for every user
    """
    euclidean_score_list = {}
    for user in users_list:
        euclidean_score_list[user] = euclidean_score(data, user1, user)

    euclidean_score_list = sorted(euclidean_score_list.items(), key=lambda x: x[1], reverse=True)

    return euclidean_score_list


def print_movies(movies, user1_movies):
    """
    Print chosen movies
    """
    count = 0
    for movie in movies:
        if count < 5 and movie not in user1_movies:
            count += 1
            print(movie)


def get_recommended_movies(data, user1, matched_user):
    """
    Choose recommended movies

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: matched_user: name of matched user
    """
    user1_movies = data[user1]
    matched_user_movies = sorted(data[matched_user].items(), key=lambda x: x[1], reverse=True)
    print("Recommended movies:")
    print_movies(matched_user_movies, user1_movies)


def get_not_recommended_movies(data, user1, scores_list):
    """
    Choose not recommended movies

    :param: data: contains users, movies and ratings
    :param: user1: name of user1
    :param: scores_list: computed movies scores
    """
    user1_movies = data[user1]
    not_recommended_movies = data[scores_list[2][0]]
    not_recommended_movies.update(data[scores_list[1][0]])
    not_recommended_movies.update(data[scores_list[0][0]])
    not_recommended_movies = sorted(not_recommended_movies.items(), key=lambda x: x[1])
    print("\nNot recommended movies:")
    print_movies(not_recommended_movies, user1_movies)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user
    ratings_file = 'ratings.json'

    with open(ratings_file, 'r', encoding='UTF8') as f:
        data = json.loads(f.read())

    users_list = get_users_list(data)
    euclideanScoreList = get_matching_results(data, user1, users_list)
    get_recommended_movies(data, user1, euclideanScoreList[0][0])
    get_not_recommended_movies(data, user1, euclideanScoreList)
