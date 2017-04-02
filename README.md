# Machine learning VU

This project contains all code and scripts for the San Fransisco crime classification challenge from Kaggle. The Challenge can be found here: https://www.kaggle.com/c/sf-crime.

## Setup your development machine

Before continuing, make sure pip is installed. Pip is the most used package manager for Python.

Install the following packages:
- **numpy** 1.12.0: *pip install numpy*
- **pandas** 0.19.2: *pip install pandas*
- **sklearn** 0.0: *pip install sklearn*

Also make sure to use Python version 2.7.13. There is no guarantee the application works on other versions.

Also, it's important that python is executable from your shell. Just type `python` and check whether it's installed.

## Datasets

To transform the original training-set provided by kaggle, run the following command: `python data_preparation $data_set $output_csv_path`. An example execution for the training-set (which can also be the test-set) can be: `python data_preparation train.csv training-set.csv`.

## Running a model

There are four models implemented for this research. The final models which resulted in the highest accuracy are called: knn_final_model.py, rf_final_model.py, neural_network_final_model.py and naive_bayes_final_model.py.

To run one of these, run the following command: `python $name_of_model_file`.

Running this will result in a print to the console of the accuracy.

## Transforming to kaggle

The two best models (KNN and RandomForest) are submittable to Kaggle. To submit, run the following command in the terminal: `python predict_test_set_with_knn.py` or `python predict_test_set_with_rf.py`. It is required for these scripts that 2 csv files are present locally. 1. is the original test set from kaggle called `test.csv` and 2. is that there is a transformed test set present called `test-set.csv`. In datasets, an explanation is given on how to transform the test set to a transformed test set.

After running the predict_test_set_with_rf, a Category column is added to the output csv that is generated. This script can deal as input in the `map_test_set_to_submissable_form.py` script.

to get the submissable test set, run the following command: `python map_test_set_to_submissable_form.py $inputfile $outputfile`. $inputfile is the csv file you want to transform, $outputfile is the name and location of the output file.

## Results

The results directory contains several txt files which shows how we did parameter tweaking and optimalization. It also has results of many runs of different models.
