#! /usr/bin/env python

import os
import pickle
import numpy as np

from vbdiar.utils.utils import Utils


class Ivec(object):
    """ Class for basic i-vector operations.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.data = None
        self.features = None
        self.window_start = None
        self.window_end = None


class IvecSet(object):
    """ Class for encapsulating ivectors set.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.name = None
        self.num_speakers = None
        self.ivecs = []

    def __iter__(self):
        current = 0
        while current < len(self.ivecs):
            yield self.ivecs[current]
            current += 1

    def __getitem__(self, key):
        return self.ivecs[key]

    def __setitem__(self, key, value):
        self.ivecs[key] = value

    def size(self):
        """ Get size of i-vector set.

        """
        return len(self.ivecs)

    def get_all(self):
        """ Get all ivectors.

        """
        a = []
        for i in self.ivecs:
            a.append(i.data.flatten())
        return np.array(a)

    def get_longer(self, min_length):
        """ Get i-vectors extracted from longer segments than minimal length.

        Args:
            min_length (int): minimal length of segment in miliseconds

        Returns:
            np.array: i-vectors
        """
        a = []
        for ivec in self.ivecs:
            if ivec.window_end - ivec.window_start >= min_length:
                a.append(ivec.data.flatten())
        return np.array(a)

    def add(self, data, window_start, window_end, mfccs=None):
        """ Add ivector to set.

            :param data: i-vector data
            :type data: numpy.array
            :param window_start: start of the window [ms]
            :type window_start: int
            :param window_end: end of the window [ms]
            :type window_end: int
        """
        i = Ivec()
        i.data = data
        i.window_start = window_start
        i.window_end = window_end
        i.features = mfccs
        self.__append(i)

    def __append(self, ivec):
        """ Append ivec to set of ivecs.

            :param ivec: input ivector
            :type ivec: Ivec
        """
        ii = 0
        for vp in self.ivecs:
            if vp.window_start > ivec.window_start:
                break
            ii += 1
        self.ivecs.insert(ii, ivec)

    def save(self, path):
        Utils.mkdir_p(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
