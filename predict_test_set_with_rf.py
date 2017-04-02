import pandas as pd
import numpy as np
import category_encoder
import rf_final_model
import sys

transformed_test_set = pd.read_csv('test-set.csv')
original_test_set = pd.read_csv('test.csv')

test_data = transformed_test_set.iloc[:,1:]

print "before predicting results"

results = rf_final_model.predict(test_data)

print "after predicting results"
original_test_set['Category'] = original_test_set.apply(lambda row: category_encoder.transform_number_to_category(results[row["Id"]]), axis=1)

print "before writing csv to file"
original_test_set.to_csv('test_set_to_submit_for_rf.csv', index=False)
