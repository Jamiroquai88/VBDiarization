#! /usr/bin/env python

import abc

from sklearn.linear_model import LogisticRegression


class Classifier(object):
    """ Class classifier creates basic interface
        for other classifiers.

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, x, y):
        """ Train classifier.

            :param x: input data
            :type x: numpy.ndarray
            :param y: input vector for class specification
            :type y: numpy.ndarray
        """
        return

    @abc.abstractmethod
    def test(self, x):
        """ Test given data with classifier.

            :param x: input data
            :type x: numpy.ndarray
            :returns: predicted class
            :rtype: numpy.ndarray
        """
        return


class LogisticRegressionModel(Classifier):
    """ Class handles operations with log. regression.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.scikit = LogisticRegression(verbose=1)

    def train(self, x, y):
        """ Fit to data.

            :param x: input data
            :type x: numpy.ndarray
            :param y: input vector for class specification
            :type y: numpy.ndarray
        """
        self.scikit.fit(x, y)

    def test(self, x, prob=False, log_prob=False):
        """ Test given data with classifier.

            :param x: input data
            :type x: numpy.ndarray
            :param prob: return probability
            :type prob: numpy.ndarray
            :param log_prob: return log of probability
            :type log_prob: numpy.ndarray
            :returns: predicted class
            :rtype: numpy.ndarray
        """
        if log_prob:
            return self.scikit.predict_log_proba(x)
        elif prob:
            return self.scikit.predict_proba(x)
        else:
            return self.scikit.predict(x)
