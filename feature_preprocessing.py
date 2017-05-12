import cPickle as pickle
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist

import Constants
import token_features
import utils
from pdf_utils.parse_docs_sax import parse_doc


def get_neighbouring_features(line_features, token_index):
    """
    Get the neighbouring features for a token from the same line
    :param line_features: features of all the tokens in the same line
    :param token_index: index of the target token
    :return:
    """
    neighbouring_features = []
    if token_index > 0:
        neighbouring_features.extend(line_features[token_index - 1])
    else:
        neighbouring_features.extend([0] * Constants.features_per_token)
    neighbouring_features.extend(line_features[token_index])
    if token_index < len(line_features) - 1:
        neighbouring_features.extend(line_features[token_index + 1])
    else:
        neighbouring_features.extend([0] * Constants.features_per_token)

    return neighbouring_features


def combine_features(previous, current, next):
    """
    Combine the features from the previous and next lines based on geometric position
    :param previous: features of previous line
    :param current: features of the target line
    :param next: features of the next line
    :return:
    """
    previous_distance = None
    next_distance = None

    if previous[1] is not None:
        previous_distance = cdist(np.array(current[1]), np.array(previous[1]))
    if next[1] is not None:
        next_distance = cdist(np.array(current[1]), np.array(next[1]))

    features = []

    for current_token_index, current_token in enumerate(current[0]):
        combined_current_feature = get_neighbouring_features(current[0], current_token_index)
        features.append(combined_current_feature)
        if previous_distance is not None:
            combined_previous_features = get_neighbouring_features(previous[0],
                                                                   previous_distance[current_token_index].argmin())
            features[current_token_index].extend(combined_previous_features)
        else:
            features[current_token_index].extend(previous[0] * 3)
        if next_distance is not None:
            combined_next_features = get_neighbouring_features(next[0], next_distance[current_token_index].argmin())
            features[current_token_index].extend(combined_next_features)
        else:
            features[current_token_index].extend(next[0] * 3)

    return features

    pass


def get_features(data, boundaries):
    """
    Extract the token based features and combine them with the neighbours' features
    :param data: data for a particular line in the pdf
    :param boundaries: the X and Y coordinate boundaries, on which the binning of position will be defined
    :return:
    """
    features = []

    if not data[0] == 0:
        bins_x = np.linspace(boundaries[0][0], boundaries[1][0], num=Constants.number_bins)
        bins_y = np.linspace(boundaries[0][1], boundaries[1][1], num=Constants.number_bins)

        centerpoints_x = []
        centerpoints_y = []

        for centerpoint in data[1]:
            centerpoints_x.append(centerpoint[0])
            centerpoints_y.append(centerpoint[1])

        binned_centerpoints_x = np.digitize(centerpoints_x, bins_x)
        binned_centerpoints_y = np.digitize(centerpoints_y, bins_y)

        for token_index, token in enumerate(data[0]):
            # lexical and dictionary features
            local_feature = token_features.get_token_feature(token)
            # width
            local_feature.append(data[3][token_index])
            # height
            local_feature.append(data[2][token_index])
            # width:height ratio
            if data[2][token_index]:
                local_feature.append(data[3][token_index] / data[2][token_index])
            else:
                local_feature.append(0)
            # one-hot bin of centroid_x
            local_feature.extend(
                utils.get_one_hot_representation(binned_centerpoints_x[token_index], Constants.number_bins))
            # one-hot bin of centroid_y
            local_feature.extend(
                utils.get_one_hot_representation(binned_centerpoints_y[token_index], Constants.number_bins))
            features.append(local_feature)
    else:
        # previous and next lines of first and last lines
        features = [0] * Constants.features_per_token
    return features

    pass


def extract_and_save_features(data, features_pickle_files):
    """

    :param data: Data from the .cxml file
    :param features_pickle_files: tuple of pickle file objects to which the extracted features/labels are to be saved at
    :return:
    """
    lines = data[0]
    centerpoints = data[1]
    heights = data[2]
    widths = data[3]
    labels = data[4]
    boundaries = data[5]

    token_labels = []

    previous_data = (0, None, 0, 0, 0)
    current_data = (lines[0], centerpoints[0], heights[0], widths[0], labels[0])
    next_data = (lines[1], centerpoints[1], heights[1], widths[1], labels[1])
    token_labels.extend(labels[0])
    token_labels.extend(labels[1])

    previous_features = get_features(previous_data[:4], boundaries)
    
    current_features = get_features(current_data[:4], boundaries)
    next_features = get_features(next_data[:4], boundaries)

    page_tokens_features = combine_features((previous_features, previous_data[1]), (current_features, current_data[1]),
                                            (next_features, next_data[1]))
    previous_features = current_features
    previous_centerpoints = current_data[1]
    current_features = next_features
    current_centerpoints = next_data[1]

    for line_index in xrange(2, len(lines)):
        next_data = (
            lines[line_index], centerpoints[line_index], heights[line_index], widths[line_index],
            labels[line_index])
        token_labels.extend(labels[line_index])
        next_features = get_features(next_data[:4], boundaries)
        page_tokens_features.extend(combine_features((previous_features, previous_centerpoints),
                                                     (current_features, current_centerpoints),
                                                     (next_features, next_data[1])))
        previous_features = current_features
        previous_centerpoints = current_centerpoints
        current_features = next_features
        current_centerpoints = next_data[1]

    next_data = (0, None, 0, 0, 0)
    next_features = get_features(next_data[:4], boundaries)
    page_tokens_features.extend(combine_features((previous_features, previous_centerpoints),
                                                 (current_features, current_centerpoints),
                                                 (next_features, next_data[1])))

    token_labels = utils.get_token_class_labels(token_labels)
    print len(token_labels)
    print np.array(page_tokens_features).shape
    pickle.dump(page_tokens_features, features_pickle_files[0])
    pickle.dump(token_labels, features_pickle_files[1])


def main(directory_path, target_libsvm_path):

    """
    :param directory_path: the path where the .cxml files of the GROTOAP dataset are saved
    :param target_libsvm_path: the path where the final extracted features are to be stored in the libsvm format
    :return: No return object

    The data, and labels extracted into the pickle files are saved in libsvm format in the path given in commandline

    """
    features_data_pickle_file = open(Constants.features_data_pickle_file_name, 'wb')
    features_labels_pickle_file = open(Constants.features_labels_pickle_file_name, 'wb')
    for pdf_xml in os.listdir(directory_path):
        if '.cxml' not in pdf_xml:
            print pdf_xml

            """
            Extract the data from the .cxml file (in Truviz format) into an object heirarchically storing the information
            """
            pdf_data = parse_doc(directory_path + utils.get_dir_separator() + pdf_xml)

            lines = []
            centerpoints = []
            heights = []
            widths = []
            labels = []
            max_x, max_y = 0, 0
            min_x, min_y = 0, 0

            for zone in pdf_data.pages[0].zones:
                for line in zone.lines:
                    local_lines = []
                    local_centerpoints = []
                    local_heights = []
                    local_widths = []
                    local_labels = []
                    for word in line.words:
                        local_lines.append(word.text)
                        local_centerpoints.append(word.centerpoint())
                        local_heights.append(word.height())
                        local_widths.append(word.width())
                        local_labels.append(word.label)
                    lines.append(local_lines)
                    centerpoints.append(local_centerpoints)
                    heights.append(local_heights)
                    widths.append(local_widths)
                    labels.append(local_labels)
                    if max_x < local_centerpoints[-1][0]:
                        max_x = local_centerpoints[-1][0]
                    if min_x > local_centerpoints[-1][0]:
                        min_x = local_centerpoints[-1][0]

            min_y = centerpoints[0][0][1]
            max_y = centerpoints[-1][0][1]

            """
            Extract token, and neighbour level features, and store them in the pickle files initialized before
            """
            extract_and_save_features((lines, centerpoints, heights, widths, labels, ((min_x, min_y), (max_x, max_y))),
                                      (features_data_pickle_file, features_labels_pickle_file))

    features_data_pickle_file.close()
    features_labels_pickle_file.close()

    """
    Extract the data from the pickle files, and save them in libsvml format
    """
    utils.save_features_as_libsvm(target_libsvm_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
