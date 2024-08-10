#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


# In[129]:


teams = pd.read_csv("teams.csv")#load data
teams


# In[130]:


teams = teams.drop(['events','height','weight','prev_3_medals'], axis =1) #Drop unnecessary columns
teams


# In[131]:


teams.corr(numeric_only="true")["medals"] #See which columns are good predictors


# In[132]:


sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)#Visualise correlation


# In[133]:


teams = teams.dropna()#remove columns with missing prev medals value


# In[134]:


train_set = teams[teams["year"] < 2012].copy() #Create training set
test_set = teams[teams["year"] >= 2012].copy() #Create test set


# In[135]:


reg = LinearRegression() 


# In[136]:


predictors = ['athletes','prev_medals'] #Specify which columns to use as predictors
target = 'medals' #Specify which columns to use as the target


# In[137]:


reg.fit(train_set[predictors],train_set[target]) #Train model using specified columns in training set


# In[138]:


predictions = reg.predict(test_set[predictors]) #Use model to predict test set


# In[139]:


predictions[predictions < 0] = 0 #Set all negative predictions to zero


# In[140]:


predictions = predictions.round() #Round all prediction to a whole number


# In[141]:


test_set_target = np.array(test_set[target]) 


# In[142]:


test_set_target = test_set_target.astype(float)
predictions = predictions.astype(float)


# In[143]:


accuracy_score(test_set_target, predictions) #Calculate accuracy score


# In[144]:


MA_error = mean_absolute_error(test_set_target, predictions) #Calculate the mean absolute error
MA_error #Check error is below standard deviation


# In[145]:


teams.describe()['medals'] #Check error is below standard deviation


# In[146]:


test_set['Predictions'] = predictions #Add predictions as a column to test set


# In[147]:


abs_error = (test_set[target]-predictions).abs() #Calculate absolute error
abs_error


# In[148]:


error_by_team = abs_error.groupby(test_set['team']).mean() #Group the absolute error by olympic teams 
error_by_team


# In[149]:


medals_by_team = test_set['medals'].groupby(test_set['team']).mean() #Group the medals by olympic teams 
medals_by_team


# In[150]:


error_ratio = error_by_team / medals_by_team #Calculate error ration
error_ratio


# In[151]:


error_ratio[~pd.isnull(error_ratio)] #Remove any null error ratios 
error_ratio = error_ratio[np.isfinite(error_ratio)] #Remove any error ratios equal to infinity


# In[152]:


error_ratio


# In[153]:


error_ratio.plot.hist() #Plot the error ration in a histogram


# In[ ]:




