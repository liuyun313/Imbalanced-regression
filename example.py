# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:10:31 2024

@author: 58211
"""


import pandas as pd
import os
import numpy as np
from importance_weighting import importance_weighting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# read data        
directory = os.getcwd()

f = 'baseball.csv'

data = pd.read_csv(f)
        
X, target = data.iloc[:,:-1], data.iloc[:,-1] 
target = np.array((target - min(target)) / (max(target) - min(target)))


# normalization and split
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, target, 
                                                    test_size=0.2, 
                                                    random_state=0)  

# 
importance_type = 'upper' # 'lower', 'both'

rg = 'lr' # lr, gbr, rf, nn

prediction = importance_weighting(X_train, X_test, y_train, y_test, importance_type, rg)      


