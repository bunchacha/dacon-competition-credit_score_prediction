import numpy as np
import pandas as pd

def create_id(input):
  data = input.copy()
  id = pd.Series(np.full((data.shape[0],), ''), index=data.index)

  # ID Containing 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'income_total', 'family_type'
  for index in data.index:
    id[index] = data.loc[index, 'family_type'] + str(data.loc[index, 'DAYS_BIRTH']) + str(data.loc[index, 'DAYS_EMPLOYED']) + str(data.loc[index, 'income_total'])
  return id

def add_features(train_df, test):
  data = pd.concat([train_df, test], axis=0)

  # Save Each Set Index
  train_idx = train_df.index
  test_idx = test.index

  # Adding Features
  cards = pd.Series(np.zeros((data.shape[0], ), dtype=int), index=data.index)
  period = pd.Series(np.zeros((data.shape[0], ), dtype=int), index=data.index)
  pcredit = pd.Series(np.zeros((data.shape[0], )), index=data.index)
  reissue = pd.Series(np.zeros((data.shape[0], ), dtype=int), index=data.index)
  sids = pd.Series(np.zeros((data.shape[0], )), index=data.index)
  
  ids = create_id(data)
  
  for id in ids.unique():
    sorted_idx = data.loc[ids[ids == id].index, ['begin_month', 'credit']].sort_values(['begin_month', 'credit']).index.tolist()
    
    for i, idx in enumerate(sorted_idx):
      if i == 0:
        sid = data.loc[idx, 'gender'] + str(idx).zfill(5)
        cards[idx] = 1
        pre_card = 1
        pre_month = data.loc[idx, 'begin_month']
        if idx in train_idx:
          pre_credit = data.loc[idx, 'credit']
        else:
          pre_credit = 1
      else:
        reissue[idx] = 1
        period[idx] = data.loc[idx, 'begin_month'] - pre_month
        pre_month = data.loc[idx, 'begin_month']

        pcredit[idx] = pre_credit
        if idx in train_idx:
          pre_card += 1
          cards[idx] = pre_card
          pre_credit = data.loc[idx, 'credit']
        else:
          cards[idx] = pre_card
      sids[idx] = sid

  result = pd.concat([cards, reissue, period, pcredit, sids], axis=1)
  result.columns = ['cards', 'reissue', 'issue_period', 'p_credit', 'ID']
  return result

drive_path = '/Users/jongseob/Documents/GitHub/dacon-competition-credit_score_prediction/data/'
raw_train = pd.read_csv(drive_path+'train.csv', index_col='index')
raw_test = pd.read_csv(drive_path+'test.csv', index_col='index')

result = add_features(raw_train, raw_test)
print(result.head(3))