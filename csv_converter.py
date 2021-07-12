import csv
import pandas as pd
import numpy as np

data = pd.read_csv('path/to/csv') 


for column in data.columns:
    if column == 'Throw-in' or column == 'Completed Passes':
        del data[column]
    else:
        continue


data['Ball Possession'] = data['Ball Possession'].str.strip('%')
data['Ball Possession'] = data['Ball Possession'].astype(int)

cols = list(data.columns)
a, b = cols.index('Team'), cols.index('Goals')
cols[b], cols[a] = cols[a], cols[b]
data = data[cols]


data.to_csv('path/to/csv')