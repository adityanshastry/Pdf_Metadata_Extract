import cPickle as pickle
import os
import platform

import numpy as np
import simstring
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import f1_score

import Constants


def get_dir_separator():
    """
    :return: the directory separator based on the operating system program is being run on
    """
    if platform.system().lower() == 'windows':
        return '\\'
    return '/'


def create_simstring_databases():
    """
    Create the simstring databases using input files in a directory
    :return:
    """
    ood_path = ".." + get_dir_separator() + "Data" + get_dir_separator() + "dicts"

    for dicts_file in os.listdir(ood_path):

        file_name = dicts_file.split(".")[0]

        if len(file_name.strip()) > 0:
            simstring_db = simstring.writer(file_name + '.db')

            for dict_word in open(ood_path + get_dir_separator() + dicts_file, 'r').readlines():
                simstring_db.insert(dict_word.strip())

            simstring_db.close()


def get_one_hot_representation(value, length):
    """

    :param value: the discrete value to be converted to one hot representation
    :param length: the length of the one hot vector
    :return: the on hot representation of the input value

    Eg. get_one_hot_representation(3, 5) returns [0, 0, 1, 0, 0]

    """
    one_hot_value = np.zeros(shape=length)
    one_hot_value[value - 1] = 1
    return list(one_hot_value)


def get_token_class_labels(token_labels):
    """
    Assign an integer to the token label as the index of the array Constants.header_classes. If the label is not present
    in the list, then it is classified as "others"
    :param token_labels:
    :return:
    """
    token_class_labels = []
    for token_label in token_labels:
        if str(token_label).lower() in Constants.header_classes:
            target_token_label = str(token_label).lower()
        else:
            target_token_label = Constants.others_class
        token_class_labels.append(Constants.header_classes.index(target_token_label) + 1)
    return token_class_labels


def save_features_as_libsvm(libsvm_file_path):
    """
    Reads the data, and labels saved in pickle files and saves them in a libsvm format in a file
    :param libsvm_file_path: path for the target libsvm file
    :return:
    """
    data, labels = [], []
    pickle_file_obj = open(Constants.features_data_pickle_file_name, "rb")
    while True:
        try:
            data.extend(pickle.load(pickle_file_obj))
        except EOFError:
            break
    pickle_file_obj.close()
    pickle_file_obj = open(Constants.features_labels_pickle_file_name, "rb")
    while True:
        try:
            labels.extend(pickle.load(pickle_file_obj))
        except EOFError:
            break

    dump_svmlight_file(X=data, y=labels, f=libsvm_file_path, zero_based=Constants.libsvm_zero_based)


def compute_f1_score(ytrue, ypred):
    """
    Given the true label values, and the predicted label values, compute the f1_score metrics
    :param ytrue: numpy array of true labels
    :param ypred: numpy array of predicted labels
    :return: dictionary with f1_score metrics for each label
    """
    tag_level_metrics = dict()

    for tag in Constants.header_classes:
        ids = np.where(ytrue == Constants.header_classes.index(tag))[0]
        if len(ids) == 0:
            continue
        yt = np.zeros(len(ytrue))
        yp = np.zeros(len(ytrue))
        yt[ids] = 1
        yp[np.where(ypred == Constants.header_classes.index(tag))] = 1

        tp = np.dot(yp, yt)
        fn = len(ids) - tp
        fp = sum(yp[np.setdiff1d(np.arange(len(ytrue)), ids)])

        if tp == 0:
            tag_level_metrics[tag] = (0, 0, 0)
        else:
            p = tp * 1. / (tp + fp)
            r = tp * 1. / (tp + fn)
            f1 = 2. * p * r / (p + r)
            tag_level_metrics[tag] = (p, r, f1)

    return tag_level_metrics


def get_test_metrics(ytrue, ypred):
    """
    Get all the metrics for the true labels, and predicted labels
    :param ytrue: numpy array of true labels
    :param ypred: numpy array of predicted labels
    :return:
    """
    accuracy = sum(ytrue == ypred) * 1. / len(ytrue)
    print 'Accuracy: {}'.format(accuracy)
    print '\n'

    w_f1 = f1_score(ytrue, ypred, average='weighted')
    print 'Weighted F1: {}'.format(w_f1)
    print '\n'

    tag_level_metrics = compute_f1_score(ytrue, ypred)

    for tag in tag_level_metrics:
        print 'Precision, Recall, F1 for ' + str(tag) + ': ' + str(tag_level_metrics[tag][0]) + ', ' + str(
            tag_level_metrics[tag][1]) + ', ' + str(tag_level_metrics[tag][2])
    print '\n'

    pass
