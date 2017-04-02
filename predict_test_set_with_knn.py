import pandas as pd
import numpy as np
import category_encoder
import knn_final_model
import sys

# Read in test set
transformed_test_set = pd.read_csv('test-set.csv')
original_test_set = pd.read_csv('test.csv')

test_data = transformed_test_set.as_matrix(knn_final_model.columns)

print "before predicting results"

# Predict the testset
results = knn_final_model.predict(test_data)

print "after predicting results"

# Create a new column and decode the predicted number to the corresponding category
original_test_set['Category'] = original_test_set.apply(lambda row: category_encoder.transform_number_to_category(results[row["Id"]]), axis=1)

print "before writing csv to file"
original_test_set.to_csv('test_set_to_submit.csv', index=False)
