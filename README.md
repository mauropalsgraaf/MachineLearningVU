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

This project requires that there is a `train.csv` file (containing the training set data) in the root of the directory with the exact name.

There is script added called `data_transformation.py` which can transform the `train.csv` to something useful.

## How to run Python scrips

To run python scripts, you can make use of the shell by typing the following command: `python $filename`. To make sure it works, I advice you to make sure you are currently in this repository directory, since file system files are loaded in.
