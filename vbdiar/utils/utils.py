#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <xprofa00@stud.fit.vutbr.cz>
# All Rights Reserved

import os
import re
import random
from os import listdir
from os.path import isfile, join
import fnmatch
import math

import numpy as np
import yaml


class Utils(object):
    """ Class tools handles basic operations with files and directories.

    """

    def __init__(self):
        """ tools class constructor.

        """
        return

    @staticmethod
    def list_directory_by_suffix(directory, suffix):
        """ Return listed directory of files based on their suffix.

            :param directory: directory to be listed
            :type directory: str
            :param suffix: suffix of files in directory
            :type suffix: str
            :returns: list of files specified byt suffix in directory
            :rtype: list

            >>> Utils.list_directory_by_suffix('../../tests/tools', '.test')
            ['empty1.test', 'empty2.test']
            >>> Utils.list_directory_by_suffix('../../tests/tools_no_ex', '.test')
            Traceback (most recent call last):
                ...
            toolsException: [listDirectoryBySuffix] No directory found!
            >>> Utils.list_directory_by_suffix('../../tests/tools', '.py')
            []
        """
        abs_dir = os.path.abspath(directory)
        try:
            ofiles = [f for f in listdir(abs_dir) if isfile(join(abs_dir, f))]
        except OSError:
            raise ValueError('No directory named {} found!'.format(directory))
        out = []
        for file_in in ofiles:
            if file_in.find(suffix) != -1:
                out.append(file_in)
        out.sort()
        return out

    @staticmethod
    def list_directory(directory):
        """ List directory.

            :param directory: directory to be listed
            :type directory: str
            :returns: list with files in directory
            :rtype: list

            >>> Utils.list_directory('../../tests/tools')
            ['empty1.test', 'empty2.test', 'test', 'test.txt']
            >>> Utils.list_directory('../../tests/tools_no_ex')
            Traceback (most recent call last):
                ...
            toolsException: [listDirectory] No directory found!
        """
        directory = os.path.abspath(directory)
        try:
            out = [f for f in listdir(directory)]
        except OSError:
            raise ValueError('No directory found!')
        out.sort()
        return out

    @staticmethod
    def recursively_list_directory_by_suffix(directory, suffix):
        """ Return recursively listed directory of files based on their suffix.

            :param directory: directory to be listed
            :type directory: str
            :param suffix: suffix of files in directory
            :type suffix: str
            :returns: list of files specified by suffix in directory
            :rtype: list

            >>> Utils.recursively_list_directory_by_suffix( \
                '../../tests/tools', '.test')
            ['empty1.test', 'empty2.test', 'test/empty.test']
            >>> Utils.recursively_list_directory_by_suffix( \
                '../../tests/tools_no_ex', '.test')
            []
        """
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*' + suffix):
                app = os.path.join(root, filename).replace(directory + '/', '')
                matches.append(app)
        matches.sort()
        return matches

    @staticmethod
    def sed_in_file(input_file, regex1, regex2):
        """ Replace in input file by regex.

            :param input_file: input file
            :type input_file: str
            :param regex1: regular expression 1
            :type regex1: str
            :param regex2: regular expression 2
            :type regex2: str
        """
        with open(input_file, 'r') as sources:
            lines = sources.readlines()
        with open(input_file, 'w') as sources:
            for line in lines:
                sources.write(re.sub(regex1, regex2, line))

    @staticmethod
    def remove_lines_in_file_by_indexes(input_file, lines_indexes):
        """ Remove specified lines in file.

            :param input_file: input file name
            :type input_file: str
            :param lines_indexes: list with lines
            :type lines_indexes: list
        """
        with open(input_file, 'r') as sources:
            lines = sources.readlines()
        with open(input_file, 'w') as sources:
            for i in range(len(lines)):
                if i not in lines_indexes:
                    sources.write(lines[i])

    @staticmethod
    def get_method(instance, method):
        """ Get method pointer.

            :param instance: input object
            :type instance: object
            :param method: name of method
            :type method: str
            :returns: pointer to method
            :rtype: method
        """
        try:
            attr = getattr(instance, method)
        except AttributeError:
            raise ValueError('Unknown class method!')
        return attr

    @staticmethod
    def configure_instance(instance, input_list):
        """ Configures instance base on methods list.

            :param instance: reference to class instance
            :type instance: object
            :param input_list: input list with name of class members
            :type input_list: list
            :returns: configured instance
            :rtype: object
        """
        for line in input_list:
            variable = line[:line.rfind('=')]
            value = line[line.rfind('=') + 1:]
            method_callback = Utils.get_method(instance, 'Set' + variable)
            method_callback(value)
        return instance

    @staticmethod
    def sort(scores, col=None):
        """ Sort scores list where score is in n-th-1 column.

            :param scores: scores list to be sorted
            :type scores: list
            :param col: index of column
            :type col: int
            :returns: sorted scores list
            :rtype: list

            >>> Utils.sort([['f1', 'f2', 10.0], \
                            ['f3', 'f4', -10.0], \
                            ['f5', 'f6', 9.58]], col=2)
            [['f3', 'f4', -10.0], ['f5', 'f6', 9.58], ['f1', 'f2', 10.0]]
            >>> Utils.sort([4.59, 8.8, 6.9, -10001.478])
            [-10001.478, 4.59, 6.9, 8.8]
        """
        if col is None:
            return sorted(scores, key=float)
        else:
            return sorted(scores, key=lambda x: x[col])

    @staticmethod
    def reverse_sort(scores, col=None):
        """ Reversively sort scores list where score is in n-th column.

            :param scores: scores list to be sorted
            :type scores: list
            :param col: number of columns
            :type col: int
            :returns: reversively sorted scores list
            :rtype: list

            >>> Utils.reverse_sort([['f1', 'f2', 10.0], \
                                   ['f3', 'f4', -10.0], \
                                   ['f5', 'f6', 9.58]], col=2)
            [['f1', 'f2', 10.0], ['f5', 'f6', 9.58], ['f3', 'f4', -10.0]]
            >>> Utils.reverse_sort([4.59, 8.8, 6.9, -10001.478])
            [8.8, 6.9, 4.59, -10001.478]
        """
        if col is None:
            return sorted(scores, key=float, reverse=True)
        else:
            return sorted(scores, key=lambda x: x[col], reverse=True)

    @staticmethod
    def get_nth_col(in_list, col):
        """ Extract n-th-1 columns from list.

            :param in_list: input list
            :type in_list: list
            :param col: column
            :type col: int
            :returns: list only with one column
            :rtype: list

            >>> Utils.get_nth_col([['1', '2'], ['3', '4'], ['5', '6']], col=1)
            ['2', '4', '6']
            >>> Utils.get_nth_col([['1', '2'], ['3', '4'], ['5', '6']], col=42)
            Traceback (most recent call last):
                ...
            toolsException: [getNthCol] Column out of range!
        """
        try:
            out = [row[col] for row in in_list]
        except IndexError:
            raise ValueError('Column out of range!')
        return out

    @staticmethod
    def find_in_dictionary(in_dict, value):
        """ Find value in directory whose items are lists and return key.

            :param in_dict: dictionary to search in
            :type in_dict: dict
            :param value: value to find
            :type value: any
            :returns: dictionary key
            :rtype: any

            >>> Utils.find_in_dictionary({ 0 : [42], 1 : [88], 2 : [69]}, 69)
            2
            >>> Utils.find_in_dictionary(dict(), 69)
            Traceback (most recent call last):
                ...
            toolsException: [findInDictionary] Value not found!
        """
        for key in in_dict:
            if value in in_dict[key]:
                return key
        raise ValueError('Value not found!')

    @staticmethod
    def get_scores(scores, key):
        """ Get scores from scores list by key.

            :param scores: input scores list
            :type scores: list
            :param key: key to find
            :type key: list
            :returns: score if key is present in score, None otherwise
            :rtype: float

            >>> Utils.get_scores([['f1', 'f2', 10.1], ['f3', 'f4', 20.1], \
                                 ['f5', 'f6', 30.1]], ['f6', 'f5'])
            30.1
        """
        if len(key) != 2:
            raise ValueError('Unexpected key!')
        if len(scores[0]) != 3:
            raise ValueError('Invalid input list!')
        for score in scores:
            a = score[0]
            b = score[1]
            if (key[0] == a and key[1] == b) or (key[0] == b and key[1] == a):
                return score[2]
        return None

    @staticmethod
    def get_line_from_file(line_num, infile):
        """ Get specified line from file.

            :param line_num: number of line
            :type line_num: int
            :param infile: file name
            :type infile: str
            :returns: specified line, None otherwise
            :rtype: str

            >>> Utils.get_line_from_file(3, '../../tests/tools/test.txt')
            'c\\n'
            >>> Utils.get_line_from_file(10, '../../tests/tools/test.txt')
            Traceback (most recent call last):
                ...
            toolsException: [getLineFromFile] Line number not found!
        """
        with open(infile) as fp:
            for i, line in enumerate(fp):
                if i == line_num - 1:
                    return line
        raise ValueError('Line number {} not found in file.'.format(line_num, infile))

    @staticmethod
    def list2dict(input_list):
        """ Create dictionary from list in format [key1, key2, score].

            :param input_list: list to process
            :type input_list: list
            :returns: preprocessed dictionary
            :rtype: dict

            >>> Utils.list2dict([['f1', 'f2', 10.1], ['f3', 'f4', 20.1], \
                                 ['f5', 'f6', 30.1], ['f1', 'f3', 40.1]])
            {'f1 f2': 10.1, 'f5 f6': 30.1, 'f3 f4': 20.1, 'f1 f3': 40.1}
            >>> Utils.list2dict([['f1', 'f2', 10.1], ['f3', 'f4']])
            Traceback (most recent call last):
                ...
            toolsException: [list2Dict] Invalid format of input list!
        """
        dictionary = dict()
        for item in input_list:
            if len(item) != 3:
                raise ValueError('Invalid format of input list!')
            tmp_list = [item[0], item[1]]
            tmp_list.sort()
            dictionary[tmp_list[0] + ' ' + tmp_list[1]] = item[2]
        return dictionary

    @staticmethod
    def merge_dicts(*dict_args):
        """ Merge dictionaries into single one.

            :param dict_args: input dictionaries
            :type dict_args: dict array
            :returns: merged dictionaries into single one
            :rtype: dict

            >>> Utils.merge_dicts( \
                {'f1 f2': 10.1, 'f5 f6': 30.1, 'f1 f3': 40.1}, {'f6 f2': 50.1})
            {'f1 f2': 10.1, 'f5 f6': 30.1, 'f6 f2': 50.1, 'f1 f3': 40.1}
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    @staticmethod
    def save_object(obj, path):
        """ Saves object to disk.

            :param obj: reference to object
            :type obj: any
            :param path: path to file
            :type path: str
        """
        np.save(path, obj)

    @staticmethod
    def load_object(path):
        """ Loads object from disk.

            :param path: path to file
            :type path: str
        """
        np.load(path)

    @staticmethod
    def common_prefix(m):
        """ Given a list of pathnames, returns the longest prefix."

           :param m: input list
           :type m: list
           :returns: longest prefix in list
           :rtype: str
        """
        if not m:
            return ''
        s1 = min(m)
        s2 = max(m)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1

    @staticmethod
    def root_name(d):
        """ Return a root directory by name.

            :param d: directory name
            :type d: str
            :returns: root directory name
            :rtype d: str
        """
        pass

    @staticmethod
    def read_config(config_path):
        """ Read config in yaml format.

        Args:
            config_path (str): path to config file

        Returns:

        """
        with open(config_path, 'r') as ymlfile:
            return yaml.load(ymlfile)

    @staticmethod
    def l2_norm(ivecs):
        """ Perform L2 normalization.

        Args:
            ivecs (np.array): input i-vector

        Returns:
            np.array: normalized i-vectors
        """
        ret_ivecs = ivecs.copy()
        ret_ivecs /= np.sqrt((ret_ivecs ** 2).sum(axis=1)[:, np.newaxis])
        return ret_ivecs

    @staticmethod
    def cos_sim(v1, v2):
        """

        Args:
            v1 (np.array): first vector
            v2 (np.array): second vector

        Returns:

        """
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    @staticmethod
    def partition(large_list, n_sublists, shuffle=False):
        """Partition a list ``l`` into ``n`` sublists."""
        return np.array_split(large_list, n_sublists)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
