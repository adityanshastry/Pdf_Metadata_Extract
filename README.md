# pdf_metadata_extract
Extraction of metadata objects (Title, Author, Abstract, Affiliations) from pdf research papers. This code is part of an independent study for the Spring 2017 semester.  Its members are:
1) Akul Siddalingaswamy - @akuls
2) Aditya Narasimha Shastry - @adityanshastry

The objective of the project is to be able to identify and extract metadata objects from a pdf research paper, specifically medical research papers. This project was completed with mentorship from:
1) Dr. Shankar Vembu - Chan Zuckerberg Initiative
2) Prof. Andrew McCallum - University of Massachusetts, Amherst

This code acts as a baseline for the process' other models to be measured against. It reads the data represented in Truviz XML format, obtained from http://cermine.ceon.pl/grotoap2/. 

Usage:
1) Place all the desired .cxml files, from the GROTOAP directories, in a folder
2) Provide that folder's path, along with the desired target file path to the python file feature_preprocessing:
    $ python feature_preprocessing.py <cxml-directory-path> <target-libsvm-file-path>
3) Once the features for the desired train, and test directories have been created in libsvm files, provide them to the logistic_regression file for training, testing, and metrics to be presented:
    $ python logistic_regression.py <train-libsvm-feature-path> <test-libsvm-feature-path>

Miscellaneous:
1) The features extracted require simstring dictionaries to be created. The dictionaries required by the code are already present in the dicts folder. If the user needs to create more dictionaries from text files, a function create_simstring_databases has been provided in the utils folder. 
