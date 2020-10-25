import pandas as pd
import numpy as np

filename = 'data_all.csv'

df=pd.read_csv(filename, sep=',', header=None)
values = df.values
values = np.delete(values, [0,1], axis=1)
#values[:, [0, -1]] = values[:, [-1, 0]]
header = values[0]
values = np.delete(values, [0], axis=0)
total_rows = values.shape[0]
indices = np.argwhere(values.astype(float) == -1)[:,0]
removed_rows = indices.shape[0]
values = np.delete(values, indices, axis=0)
values = np.append([header], values, axis=0)
print(str(removed_rows), "/", str(total_rows), "removed")
df = pd.DataFrame(values)
df.to_csv('clean_data.csv', header=None, index=False)
