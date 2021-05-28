from preprocessing import pre_prcs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def fill_occyp(train, test, input):
  # Features (absolute value of correlation with occyp > 0.05)
  category = ['income_type', 'work_phone', 'house_type', 'edu_type', 'child_num', 'gender', 'car']
  numeric = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'income_total']

  total = pd.concat([train, test], axis=0)
  # occyp train set
  occyp_train = total[ total['occyp_type']!='None' ].loc[ :, category + numeric ].copy()
  y           = total[ total['occyp_type']!='None' ]['occyp_type'].copy()
  
  # occyp test set (fill null)
  occyp_test = input[ input['occyp_type']=='None' ][ input['DAYS_EMPLOYED']!=0 ].loc[ :, category + numeric ].copy()

  # pre-processing
  X    = pre_prcs(occyp_train, occyp_train, category, numeric)
  test = pre_prcs(occyp_train, occyp_test, category, numeric)
  
  # Create RandomForest Model and Show CV Result
  rf = RandomForestClassifier(random_state=13)
  print( f"Train Set CV Score : {cross_val_score(rf, X, y, cv=5)}")
  
  # Machine Learning
  rf.fit( X, y )

  # Test Label Prediction
  predictions = rf.predict(test)
  index = test.index
  
  # Replacement
  result = input['occyp_type'].copy()
  result[index] = predictions

  # None to Retired
  non_idx = result[ result=='None'].index
  result.loc[non_idx] = 'Retired'
  
  return result

# drive_path = '/Users/jongseob/Documents/GitHub/dacon-competition-credit_score_prediction/data/'
# raw_train = pd.read_csv(drive_path+'train.csv', index_col='index')
# raw_test = pd.read_csv(drive_path+'test.csv', index_col='index')
# raw_train.fillna('None', inplace=True, axis=0)
# raw_test.fillna('None', inplace=True, axis=0)

# result = fill_occyp(raw_train, raw_test, raw_train)
# print(result.head(3))