import numpy as np
import pandas as pd
from random import sample
import utils
# data_path = 'train_FD004.txt'
# data = utils.load_data(data_path)
# x = data.iloc[:, 2:26]
# x.insert(0, 'engine_no', data.iloc[:, 0])
# print(x[:][1:])
# a = [2,3,4]
# b = [3,5,6]
# c = []
# c.append(np.array(a)-np.array(b))
# c.append(np.array(b)-np.array(a))
# print(np.array(c))
a = np.identity(5)*0.95
print(a)
for i in range(a.shape[0] - 1):
    a[i, i+1] = 0.05
a[-1, -1] = 1
print(a)