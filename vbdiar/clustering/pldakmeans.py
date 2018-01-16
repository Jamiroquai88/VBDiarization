#! /usr/bin/env python

import numpy as np


class PLDAKMeans(object):
    """ KMeans clustering algorithm using PLDA output as distance metric.

    """
    def __init__(self, centroids, k, plda, max_iter=1000):
        """ Class constructor.

            :param centroids: initialization centroids
            :type centroids: numpy.array
            :param k: number of classes
            :type k: int
            :param plda: PLDA object
            :type plda: PLDA
            :param max_iter: maximal number of iterations
            :type max_iter: int
        """
        self.max_iter = max_iter
        self.old_labels = []
        self.data = None
        self.cluster_centers_ = centroids
        self.k = k
        self.plda = plda

    def fit(self, data):
        """ Fit the input data.

            :param data: input data
            :type data: numpy.array
            :returns: cluster centers
            :rtype: numpy.array
        """
        self.data = data
        iterations = 0
        while True:
            if self.stop(iterations):
                break
            else:
                iterations += 1
        return self.cluster_centers_

    def stop(self, iterations):
        """ Make the decision if algorithm should stop.

            :param iterations: number of successfull iterations
            :type iterations: int
            :returns: True if algorithm should stop, False otherwise
            :rtype: bool
         """
        labels = self.labels()
        if iterations > self.max_iter or self.old_labels == labels:
            return True
        else:
            self.old_labels = labels
            return False

    def labels(self):
        """ Predict labels.

        """
        scores = self.plda.score(self.data, self.cluster_centers_)
        centroids = {}
        for ii in range(self.k):
            centroids[ii] = []
        labels = []
        for ii in range(self.data.shape[0]):
            c = np.argmax(scores.T[ii])
            labels.append(c)
            centroids[c].append(self.data[ii])
        for ii in range(self.k):
            self.cluster_centers_[ii] = np.mean(centroids[ii], axis=0)
        return labels
