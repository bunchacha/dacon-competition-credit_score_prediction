import pandas as pd

def replace_class(column, before, after):
  series = column.copy()
  for cls in list(series.unique()):
    for i, b in enumerate(before):
      if cls in b:
        series.replace(cls, after[i], inplace=True)
  return series

drive_path = '/Users/jongseob/Documents/GitHub/dacon-competition-credit_score_prediction/data/'
raw_train = pd.read_csv(drive_path+'train.csv', index_col='index')

before = [['House / apartment', 'Co-op apartment'],
          ['With parents'],
          ['Rented apartment', 'Office apartment', 'Municipal apartment']]
after = ['House / apartment', 'With parents', 'Rented apartment']

house_new = replace_class(raw_train['house_type'], before, after)

print(raw_train['house_type'].head(3), house_new.head(3))
