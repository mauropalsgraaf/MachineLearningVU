import pandas as pd
import numpy as np
import category_encoder
import knn_final_model
import sys

transformed_test_set = pd.read_csv('test-set.csv')
original_test_set = pd.read_csv('test.csv')

test_data = transformed_test_set.as_matrix(knn_final_model.columns)

results = knn_final_model.predict(test_data)

original_test_set['Category'] = original_test_set.apply(lambda row: category_encoder.transform_number_to_category(row["Id"]), axis=1)

original_test_set.to_csv('test_set_to_submit.csv', index=False)
