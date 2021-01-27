"""
Module for testing the Dataset class.
"""

import random

import pytest
import pandas as pd

from tiny_clues_recommander.data import Dataset, Trainset, InfoData

random.seed(1)

books = pd.DataFrame({
    'ISBN' : ['0195153448', '0002005018', '0060973129', '0374157065', '0393045218'],
    'Book-title' : ['Classical Mythology', 'Clara Callan', 'Decision in Normandy','Flu: The Story of the Great Influenza','The Mummies'],
    'year' : [2002, 2001, 1991, 1999, 1999]
})

ratings = pd.DataFrame({
    'ISBN' : ['0195153448', '0195153448', '0195153448', '0002005018', '0002005018', '0060973129', '0374157065', '0374157065', '0393045218'],
    'User' : [1, 2, 4, 1, 3, 2, 2, 1, 4],
    'ratings' : [3, 4, 3, 5, 5, 3, 2, 1, 3]
})

users = pd.DataFrame({
    'User' : [1, 2, 3, 4, 5],
    'Name' : ['Thomas', 'Henry', 'Joe', 'Lili', 'Jonas']
})


def test_build_full_trainset():
    """Test the build_full_trainset method."""

    data = Dataset(ratings)
    trainset = data.build_trainset()

    assert len(trainset.ur) == 5
    assert len(trainset.ir) == 4
    assert trainset.n_users == 5
    assert trainset.n_items == 4

def test_iterate_over_trainset():

    data = Dataset(ratings)
    trainset = data.build_trainset()

    assert len(ratings) == len([rating for rating in trainset.all_ratings()])

    assert next(trainset.all_ratings()) == (0, 0, 3.0)


def test_mappings_ids():
    """Ensure Mappings is Ok"""

    # DF creation.
    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, '10000'],
                    'rating': [3, 2, 4, 3, 1],
                    'other': [12, 2, 11, 32, 133]}
    df = pd.DataFrame(ratings_dict)

    data = Dataset(df[['userID', 'itemID', 'rating']])

    trainset = data.build_trainset()

    # assert r(9, 1) = 3 and r(2, 1) = 4
    uid9 = trainset.to_inner_uid(9)
    uid2 = trainset.to_inner_uid(2)
    iid1 = trainset.to_inner_iid(1)
    assert trainset.ur[uid9] == [(iid1, 3)]
    assert trainset.ur[uid2] == [(iid1, 4)]


def test_info_data():
    """Ensure Info Data is Ok"""

    data = Dataset(ratings)
    trainset = data.build_trainset()
    book_info = InfoData(books, 'ISBN')
    trainset.set_item_info(book_info)
    user_info = InfoData(users, 'User')
    trainset.set_user_info(user_info)

    assert trainset.get_itemid_info('0195153448')['Book-title'] == 'Classical Mythology'
    assert trainset.get_itemid_info('0002005018')['year'] == 2001
    assert trainset.get_userid_info(1) == {'Name' : 'Thomas'}

