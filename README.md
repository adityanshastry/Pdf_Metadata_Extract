# pdf_metadata_extract

<strong> Introduction: </strong> </br>
The objective of the project is to be able to identify and extract metadata objects, namely Title, Authors, Affiliations, and Abstract, from a pdf research paper, specifically medical research papers. This code is part of an independent study project for the Spring 2017 semester.  Its members are:
1) Akul Siddalingaswamy - https://github.com/akuls/
2) Aditya Narasimha Shastry - https://github.com/adityanshastry/

This project was completed with mentorship from:
1) Dr. Shankar Vembu - Chan Zuckerberg Initiative
2) Prof. Andrew McCallum - University of Massachusetts, Amherst

This code acts as a baseline for the project's other models to be measured against. It reads the data represented in Truviz XML format, obtained from http://cermine.ceon.pl/grotoap2/. 

<strong> Usage: </strong>
1) Place all the desired .cxml files, from the GROTOAP directories, in a folder
2) Provide that folder's path, along with the desired target file path to the python file feature_preprocessing: </br>
    $ python feature_preprocessing.py cxml_directory_path target_libsvm_file_path </br>
    This step creates temporary pickle files for processing. The path for these pickle files are present in the Constants.py file with the variable names features_data_pickle_file_name, and features_labels_pickle_file_name. These need to be modified by the user as desired.
3) Once the features for the desired train, and test directories have been created in libsvm files, provide them to the logistic_regression file for training, testing, and metrics to be presented: </br>
    $ python logistic_regression.py train_libsvm_feature_path test_libsvm_feature_path

<strong> Miscellaneous: </strong>
1) The features extracted require simstring dictionaries to be created. The dictionaries required by the code are already present in the dicts folder. If the user needs to create more dictionaries from text files, a function "create_simstring_databases" has been provided in the utils folder. 
