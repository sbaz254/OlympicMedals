#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


# In[99]:


teams = pd.read_csv("teams.csv")#load data
teams


# In[100]:


teams = teams.drop(['events','height','weight','prev_3_medals'], axis =1) #Drop unnecessary columns
teams


# In[101]:


teams.corr(numeric_only="true")["medals"] #See which columns are good predictors


# In[102]:


sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)


# In[103]:


teams = teams.dropna()#remove columns with missing prev medals value


# In[104]:


train_set = teams[teams["year"] < 2012].copy()
test_set = teams[teams["year"] >= 2012].copy()


# In[105]:


reg = LinearRegression()


# In[106]:


predictors = ['athletes','prev_medals']
target = 'medals'


# In[107]:


reg.fit(train_set[predictors],train_set[target])


# In[108]:


predictions = reg.predict(test_set[predictors])


# In[109]:


predictions[predictions < 0] = 0


# In[110]:


predictions = predictions.round()


# In[111]:


test_set_target = np.array(test_set[target])
test_set_target


# In[112]:


predictions


# In[113]:


test_set_target = test_set_target.astype(float)
predictions = predictions.astype(float)


# In[114]:


accuracy_score(test_set_target, predictions)


# In[115]:


MA_error = mean_absolute_error(test_set_target, predictions)
MA_error #check error is below standard deviation


# In[116]:


teams.describe()['medals']


# In[117]:


test_set['Predictions'] = predictions


# In[118]:


test_set


# In[119]:


test_set[test_set["team"] == "USA"]


# In[120]:


abs_error = (test_set[target]-predictions).abs()
abs_error


# In[121]:


error_by_team = abs_error.groupby(test_set['team']).mean()
error_by_team


# In[122]:


medals_by_team = test_set['medals'].groupby(test_set['team']).mean()
medals_by_team


# In[123]:


error_ratio = error_by_team / medals_by_team
error_ratio


# In[124]:


error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]


# In[125]:


error_ratio


# In[126]:


error_ratio.plot.hist()


# In[ ]:




