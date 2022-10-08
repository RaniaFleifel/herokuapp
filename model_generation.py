#!/usr/bin/env python
# coding: utf-8

# In[2]:


######################################################################
#dataset from https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download

import pandas as pd
import pickle

df=pd.read_csv('Fish.csv')

#df.head()

features= df[["Length1","Length2","Length3","Height","Width"]]
target=df.loc[:,df.columns=='Weight']

#print(features)

####################################################################
# from https://github.com/ksatola/ml-introduction/blob/master/11_SimpleLinearRegression.ipynb
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#print(feature_train)
reg.fit(feature_train, target_train)

r2_test = reg.score(feature_test, target_test) # low if overfitted
r2_train = reg.score(feature_train, target_train) # this is just to compare, as we should trust the score on the test data

print("R Squared for test: {}".format(r2_test))
print("R Squared for train: {}".format(r2_train))

# Save the Model for later use
model_filename = 'fish_LinearRegression.model'
pickle.dump(reg, open(model_filename, 'wb'))
# ######################################################################
