import os
import sys

import Constants
from pdf_utils.parse_docs_sax import parse_doc


def main(directory_path, target_path):
    """
    :param directory_path: the path where the .cxml files of the GROTOAP dataset are saved
    :param target_libsvm_path: the path where the final extracted features are to be stored in the libsvm format
    :return: No return object

    The data, and labels extracted into the pickle files are saved in libsvm format in the path given in commandline

    """

    output_fp = open(target_path, "w")

    for pdf_xml in os.listdir(directory_path):
        if '.cxml' not in pdf_xml:
            # print pdf_xml

            """
            Extract the data from the .cxml file (in Truviz format) into an object heirarchically storing the information
            """
        else:
            print pdf_xml
            output_fp.write("\n0:0:612:792\n")
            pdf_data = parse_doc(directory_path + "/" + pdf_xml)

            for zone in pdf_data.pages[0].zones:
                for line in zone.lines:
                    for word in line.words:
                        label = "O"
                        if str(word.label).lower() in Constants.header_classes:
                            label = "I-" + word.label
                        output_fp.write(word.text.encode('ascii', 'ignore') + " 1:" + str(word.centerpoint()[0]) + ":" + str(
                            word.centerpoint()[1]) + ":" + str(word.height()) + ":" + str(
                            word.width()) + " * " + label + "\n")

    output_fp.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
