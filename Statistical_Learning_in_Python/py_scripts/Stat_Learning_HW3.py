import pandas as pd

data=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\Crabs.csv')
data=pd.get_dummies(data,columns=['color','spine'],drop_first=1)

import numpy as np
from sklearn.model_selection import train_test_split 

np.random.seed(2)
train, test = train_test_split(data, test_size=.3)

