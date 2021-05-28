import numpy as np
import pandas as pd

# Get Sum of Weighted Scores of Customized Standards
def score(input):
  data = input.reset_index()
  house_scorecard = {'House / apartment' : 42,
                    'With parents' : 32,
                    'Rented apartment' : 28}
  reality_scorecard = {'Y' : 40,
                      'N' : 30}
  edu_scorecard = {'Lower secondary' : 12,
                  'Secondary / secondary special' : 14,
                  'Incomplete higher' : 16,
                  'Higher education' : 18,
                  'Academic degree' : 20}
  occyp_scorecard = {'Service' : 36,
                     'Non_service' : 36,
                     'Office' : 38,
                     'Specialized' : 40,
                     'Retired' : 36,}
  income_label = [71, 79, 85, 92, 103, 111]
  age_label = [110, 104, 93, 84, 77, 69]
  employ_label = [36, 27, 20]

  points = np.zeros((data.shape[0],), dtype=int)
  for idx in data.index:
    points[idx] += house_scorecard[data.loc[idx, 'house_new']]
    # points[idx] += reality_scorecard[data.loc[idx, 'reality']]
    # points[idx] += edu_scorecard[data.loc[idx, 'edu_type']]
    points[idx] += occyp_scorecard[data.loc[idx, 'occyp_new']]

  points += np.array(pd.qcut(data['income_total'], q=6, labels=income_label).astype('int64'))
  points += np.array(pd.qcut(data['mDB'], q=6, labels=age_label).astype('int64'))
  points += np.array(pd.qcut(data['mDE'], q=4, labels=employ_label, duplicates='drop').astype('int64'))

  return pd.DataFrame(points, index=input.index)