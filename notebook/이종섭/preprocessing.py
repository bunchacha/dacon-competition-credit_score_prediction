import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

def pre_prcs(train, input, category_features, numeric_features):
  idx = input.index
  
  scaler = MaxAbsScaler()
  scaler.fit(train[numeric_features])
  numeric = pd.DataFrame( scaler.transform(input[numeric_features]), 
                          columns=numeric_features, index=idx)

  encoder = OneHotEncoder()
  encoder.fit(train[category_features])
  category = pd.DataFrame( encoder.transform(input[category_features]).toarray(), 
                           columns=encoder.get_feature_names() , index=idx ) 

  return pd.concat( [numeric, category], axis=1 )

drive_path = '/Users/jongseob/Documents/GitHub/dacon-competition-credit_score_prediction/data/'
raw_train = pd.read_csv(drive_path+'train.csv', index_col='index')
raw_train.dropna(inplace=True, axis=0)
categoty = ['gender', 'car', 'reality', 'income_type','edu_type', 'family_type', 
            'house_type', 'FLAG_MOBIL', 'work_phone', 'phone', 'email', 'occyp_type',
            'family_size', 'child_num']
numeric = ['income_total', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'begin_month']

result = pre_prcs(raw_train.iloc[:, :-1], raw_train.iloc[:, :-1], categoty, numeric)
print(result.head(3))