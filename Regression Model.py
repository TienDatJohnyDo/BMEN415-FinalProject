#!/usr/bin/env python
# coding: utf-8

# # Regression Model 
# 
# Tien Dat Johny Do -30087967

# In[24]:


#import classes into 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[25]:


#Import the data set of breast cancer for our classification data set
#create a directory and dinf the csv file
directory = './Volumetric_features.csv'
csv_file = pd.read_csv(directory)
#Drop S.No
csv_file.iloc[:,1:]

#Print out the file within coloumns
csv_file


# ### Multiple Linear Regression Model
# 
# https://www.analyticsvidhya.com/blog/2021/05/multiple-linear-regression-using-python-and-scikit-learn/
# 

# In[26]:


x= csv_file.drop('dataset',axis=1)
x = x.values


# In[27]:


y = csv_file.loc[:,['Age']]
y = y.values


# In[28]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[29]:


# importing module
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
# creating an object of LinearRegression class
LR = linear_model.LinearRegression()

# fitting the training data
LR.fit(x_train,y_train)


# In[30]:


y_prediction =  LR.predict(x_test)
#y_prediction


# In[31]:


# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print('r2 score is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))


# ### Bagged Model 
# 
# https://machinelearningmastery.com/bagging-ensemble-with-python/

# In[32]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[33]:


seed = 7
kfold = KFold(n_splits = 10,)
cart = DecisionTreeClassifier()


# In[34]:


num_trees = 150


# In[35]:


model = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed)


# In[36]:


results = cross_val_score(model, x, y.ravel(), cv=kfold)


# In[37]:


print(results.mean())


# In[38]:


print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))


# ### kNN Model
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

# In[39]:


import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor


# In[40]:


model = KNeighborsRegressor(n_neighbors=5)
print(model)
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=1, n_neighbors=5, p=2,
          weights='uniform') 


# In[41]:


model.fit(x,y)


# In[42]:


pred_y = model.predict(x)


# In[43]:


score=model.score(x,y)
print(score)


# In[44]:


mse =mean_squared_error(y, pred_y)
print("Mean Squared Error:",mse)


# In[45]:


rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

