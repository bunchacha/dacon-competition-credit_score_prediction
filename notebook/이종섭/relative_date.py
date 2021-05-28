import pandas as pd
import numpy as np

# Modify Date Information accoding to Relative Time Difference 'begin_month'
def relative_date(input1, input2):
  total = pd.concat([input1, input2], axis=0)
  index = total.index
  temp = total.reset_index()

  mdb = np.zeros((temp.shape[0], ), dtype=int)
  mde = np.zeros((temp.shape[0], ), dtype=int)

  for id in temp['ID'].unique():
    indices = temp[ temp['ID'] == id ].sort_values(['begin_month', 'credit'], axis=0).index.tolist()
    indices.reverse()

    for i, k in enumerate(indices):
      if i==0:
        first = temp.loc[k, 'begin_month']
        mdb[k] = temp.loc[k, 'DAYS_BIRTH']
        mde[k] = temp.loc[k, 'DAYS_EMPLOYED']
      else:
        mdb[k] = temp.loc[k, 'DAYS_BIRTH'] - (temp.loc[k, 'begin_month'] - first)*30
        
        if temp.loc[k, 'DAYS_EMPLOYED'] == 0:
          mde[k] = 0
        elif temp.loc[k, 'DAYS_EMPLOYED'] > (temp.loc[k, 'begin_month'] - first)*30:
          mde[k] = 0
        else:
          mde[k] = temp.loc[k, 'DAYS_EMPLOYED'] - (temp.loc[k, 'begin_month'] - first)*30
  
  result = pd.DataFrame(np.array([mdb, mde]).T, index=index, columns=['mDB', 'mDE'] )
  result1 = result.loc[input1.index, :]
  result2 = result.loc[input2.index, :]
  return result1, result2