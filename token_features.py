import re

import simstring

import Constants
import utils


def check_case(token):
    if token.islower():
        # token is all lower case
        return 1

    if token.isupper():
        # token is all upper case
        return 2

    # token is mixed case
    return 3


def check_email(token):
    if re.match(r"[^@]+@[^@]+\.[^@]+", token):
        return 1
    return 0


def check_numeric(token):
    try:
        float(token)
        # token is a number
        return 1
    except ValueError:
        if token.isalnum():
            # token is alpha-numeric
            return 2
    # token is neither
    return 0


def get_word_length(token):
    return len(token)


def check_dictionary_db(token, database):
    dict_db = simstring.reader(database)
    dict_db.measure = simstring.cosine
    dict_db.threshold = 0.9
    if len(dict_db.retrieve(token.encode('utf-8'))) > 0:
        return 1
    return 0


def get_token_feature(token):
    """
    Extract the lexical and style based features for a token
    :param token: the value of the token extracted from the .cxml file
    :return: the features for the token
    """
    feature = []

    feature.extend(utils.get_one_hot_representation(check_case(token), 3))
    feature.append(check_email(token))
    feature.extend(utils.get_one_hot_representation(check_numeric(token), 3))
    if check_dictionary_db(token, Constants.city_full_db) or check_dictionary_db(token,
                                                                                 Constants.region_full_db) or check_dictionary_db(
        token, Constants.country_full_db):
        feature.append(1)
    else:
        feature.append(0)
    if check_dictionary_db(token, Constants.department_keywords_db) or check_dictionary_db(token,
                                                                                           Constants.department_full_db) or check_dictionary_db(
        token, Constants.faculty_full_db) or check_dictionary_db(token, Constants.faculty_keywords_db):
        feature.append(1)
    else:
        feature.append(0)
    if check_dictionary_db(token, Constants.university_full_db) or check_dictionary_db(token,
                                                                                       Constants.university_keywords_db):
        feature.append(1)
    else:
        feature.append(0)
    if check_dictionary_db(token, Constants.people_names_db):
        feature.append(1)
    else:
        feature.append(0)

    return feature
