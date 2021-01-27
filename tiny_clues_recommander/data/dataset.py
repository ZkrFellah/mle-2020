from collections import defaultdict
import numpy as np
from six import iteritems

class Dataset():
    def __init__(self, df):

        self.df = df
        self.raw_ratings = [(uid, iid, float(r), None)
                                for (uid, iid, r) in
                                self.df.itertuples(index=False)]
        self.rating_scale = (df.iloc[:, 2].min(), df.iloc[:, 2].max())

    def build_trainset(self):
        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in self.raw_ratings:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(self.raw_ratings)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.rating_scale,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

class InfoData():
    """A Information Data contains all useful data that could describe Items or Users like name, age, country ...
    Attributes:
        df(pd.DataFrame): DataFrame  of MetaData
        index_col(object): the name of the index columns
    """
    def __init__(self, df, index_col):
        if df.index.name != index_col:
            self.df = df.set_index(index_col)
        else: 
            self.df = df
    
    def get_inforamtions(self, index, columns=None):
        if columns is None:
            columns = self.df.columns
        record = self.df.loc[index, columns]
        return {k:v for k,v in record.items()}


class Trainset:
    """A trainset contains all useful data that constitute a training set.
    It is used by the :meth: fit() of every
    prediction algorithm.
    Attributes:
        ur(:obj:`defaultdict` of :obj:`list`): The users ratings. This is a
            dictionary containing lists of tuples of the form ``(item_inner_id,
            rating)``. The keys are user inner ids.
        ir(:obj:`defaultdict` of :obj:`list`): The items ratings. This is a
            dictionary containing lists of tuples of the form ``(user_inner_id,
            rating)``. The keys are item inner ids.
        n_users: Total number of users :math:`|U|`.
        n_items: Total number of items :math:`|I|`.
        n_ratings: Total number of ratings :math:`|R_{train}|`.
        rating_scale(tuple): The minimum and maximal rating of the rating
            scale.
        global_mean: The mean of all ratings :math:`\\mu`.
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 raw2inner_id_users, raw2inner_id_items):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None
        # inner2raw dicts could be built right now (or even before) but they
        # are not always useful so we wait until we need them.
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None

    def set_user_info(self, user_info):
        self.user_info = user_info

    def set_item_info(self, item_info):
        self.item_info = item_info
    
    def get_userid_info(self, uid, columns=None, is_raw_id=True):
        if not is_raw_id:
            uid = self.to_raw_uid(uid)
        return self.user_info.get_inforamtions(uid, columns)

    def get_itemid_info(self, iid, columns=None, is_raw_id=True):
        if not is_raw_id:
            iid = self.to_raw_iid(iid)
        return self.item_info.get_inforamtions(iid, columns)

    def to_inner_uid(self, ruid):
        """Convert a **user** raw id to an inner id.
        Args:
            ruid(str): The user raw id.
        Returns:
            int: The user inner id.
        Raises:
            ValueError: When user is not part of the trainset.
        """

        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError('User ' + str(ruid) +
                             ' is not part of the trainset.')

    def to_raw_uid(self, iuid):
        """Convert a **user** inner id to a raw id.
        Args:
            iuid(int): The user inner id.
        Returns:
            str: The user raw id.
        Raises:
            ValueError: When ``iuid`` is not an inner id.
        """

        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_users)}

        try:
            return self._inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(str(iuid) + ' is not a valid inner id.')

    def to_inner_iid(self, riid):
        """Convert an **item** raw id to an inner id.
        Args:
            riid(str): The item raw id.
        Returns:
            int: The item inner id.
        Raises:
            ValueError: When item is not part of the trainset.
        """

        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError('Item ' + str(riid) +
                             ' is not part of the trainset.')

    def to_raw_iid(self, iiid):
        """Convert an **item** inner id to a raw id.
        Args:
            iiid(int): The item inner id.
        Returns:
            str: The item raw id.
        Raises:
            ValueError: When ``iiid`` is not an inner id.
        """

        if self._inner2raw_id_items is None:
            self._inner2raw_id_items = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_items)}

        try:
            return self._inner2raw_id_items[iiid]
        except KeyError:
            raise ValueError(str(iiid) + ' is not a valid inner id.')

    def all_ratings(self):
        """Generator function to iterate over all ratings.
        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    def all_users(self):
        """Generator function to iterate over all users.
        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.
        Yields:
            Inner id of items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean
