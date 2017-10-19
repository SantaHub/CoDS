#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:09:17 2017

@author: austin + Yashwanth
"""
# cd /home/austin/ML/LAB_ADML_temp/data

import numpy as np
import pandas as pd

data = pd.read_csv('Karunya_data.csv')

#No duplicates. But Radiation is 0 for 5518 datasets

y=np.array(data.SolRad)

features=list(data)
features.remove('SolRad')
X=data.as_matrix(features).astype(np.float)


from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
X=scaler.fit_transform(X)

from sklearn.ensemble import RandomForestRegressor

clf=RandomForestRegressor()
clf.fit(X[:30000],y[:30000])
y_pred=clf.predict(X[30000:])

from sklearn.metrics import r2_score
r2_score(y[30000:],y_pred)
