## Predicting the number of drinks. 
# data : uci bupa liver dataset.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:23:14 2018

@author: austin
"""

from sklearn import tree
import pandas as pd

data = pd.read_csv('./bupa.data')
X= data.iloc[1:,:5]
y= data.iloc[1:,6]

from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.2, random_state=42)

clf = tree.DecisionTreeRegressor()
clf = clf.fit(trainX, trainy)
pred = clf.predict(testX)

from sklearn.metrics import accuracy_score,confusion_matrix

cm  = confusion_matrix(testy, pred)
acc= accuracy_score(testy,pred)

import matplotlib.pyplot as plt
plt.plot(cm)
plt.ylabel('Confusion matrix')

plt.plot()
plt.ylabel('Accuracy score')
plt.show()
