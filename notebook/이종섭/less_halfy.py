import numpy as np
import pandas as pd

# (Binary) Whether Card Issued in less 6 months
def less_halfy(input):
  data = input.reset_index()
  new_issue = np.zeros((data.shape[0],), dtype=int)

  idx = data[data['begin_month'] > -7].index
  new_issue[idx] = 1
  return new_issue