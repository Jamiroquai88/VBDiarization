#!/usr/bin/env python

from user_exception import ClusterException


class Cluster(object):
    """ Interface for performing any kind of clustering algorithm.

    """
    def __init__(self):
        """ Class constructor.

        """
        self.clusters = {}

    def add(self, label, value):
        """ Add label and value to the clusters.

            :param label: cluster unique label
            :type label: any
            :param value: value of label
            :type value: any
        """
        self.clusters[(label,)] = [value]

    def merge(self, label1, label2):
        """ Merge two clusters.

            :param label1: label of first cluster
            :type label1: any
            :param label2: label of second cluster
            :type label2: any"""
        if label1 != label2:
            key1, values1, key2, values2 = None, None, None, None
            for key in self.clusters:
                if label1 in key:
                    key1 = key
                if label2 in key:
                    key2 = key
            if key1 is None:
                raise ClusterException(
                    '[Cluster.merge] Unexpected value of label = {}.'.format(label1)
                )
            if key2 is None:
                raise ClusterException(
                    '[Cluster.merge] Unexpected value of label = {}.'.format(label2)
                )
            if key1 != key2:
                self.clusters[key1 + key2] = self.clusters.pop(key1, None) + self.clusters.pop(key2, None)

    def size(self):
        """ Get number of clusters.

            :returns: number of clusters
            :rtype: int
        """
        return len(self.clusters.keys())


if __name__ == "__main__":
    c = Cluster()
    c.add('a', 1)
    c.add('b', 2)
    c.add('c', 3)
    c.add('d', 4)
    c.add('e', 5)
    c.add('f', 6)
    c.add('g', 7)
    c.add('h', 8)
    c.merge('a', 'b')
    c.merge('c', 'd')
    c.merge('a', 'd')
    c.merge('a', 'a')
    c.merge('a', 'b')
