import sys

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegressionCV

import Constants
import utils


def logistic_regression_search(train_file_path):
    """
    This fucntion performs Logistic Regression learning and classification
    5 Fold CV is used and the best regularization parameter is chosen using the trainig set
    :param train_file_path: the path where the training data is saved in libsvm format
    :return:
    """

    train_data, train_labels = load_svmlight_file(f=train_file_path, zero_based=Constants.libsvm_zero_based)

    lr_search = LogisticRegressionCV(Cs=list(np.power(2.0, np.arange(-15, 15))),
                                     penalty='l2', cv=5, random_state=777, max_iter=10000,
                                     fit_intercept=True, solver='liblinear', refit=True)

    lr_search.fit(train_data, train_labels)

    return lr_search


def logistic_regression_test(lr_obj, test_file_path):
    """
    Perform logistic regression testing, using the train object on the test features
    :param lr_obj: the object which contains the trained coefficients for the train data
    :param test_file_path: the path where the test data is saved in libsvm format
    :return:
    """
    test_data, test_labels = load_svmlight_file(f=test_file_path, zero_based=Constants.libsvm_zero_based)
    test_pred = lr_obj.predict(test_data)
    utils.get_test_metrics(test_labels, test_pred)


if __name__ == '__main__':
    lr_train_obj = logistic_regression_search(sys.argv[1])
    logistic_regression_test(lr_train_obj, sys.argv[2])
