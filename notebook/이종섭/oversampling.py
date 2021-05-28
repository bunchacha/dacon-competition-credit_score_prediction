import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

def oversample(input, category):
  data = input.copy()
  
  # Dimensionality Reduction (Categorical Features)
  value = 0
  hash = {}
  for row in data[ ~data.duplicated(category, keep='first') ][category].values:
    hash[','.join(row.astype(str))] = value
    value += 1
  
  numeric = [ele for ele in data.columns.tolist() if ele not in category]
  numeric.remove('credit')

  keys = pd.Series(np.zeros((data.shape[0],), dtype=int, name='key'), index=data.index)
  for idx in data.index:
    key = ','.join(data.loc[idx, category].values.astype(str))
    keys[idx] = hash[key]

  X = pd.concat([keys, data[numeric]], axis=1) #data[['key'] + numeric]
  y = data['credit']

  smote = SMOTE(random_state=13)
  X_sample, y_sample = smote.fit_resample(X, y)
  # ['key', 'income_total', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'family_size', 'begin_month']
  
  hash_t  = dict([item[::-1]for item in hash.items()])
  category_t = []
  for idx in X_sample.index:
    category_t.append(hash_t[X_sample.loc[idx, 'key']].split(','))
  
  X_sample[category] = category_t
  X_sample.drop('key', axis=1, inplace=True)

  return X_sample, y_sample

