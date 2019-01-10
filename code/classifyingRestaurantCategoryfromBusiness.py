#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


# In[2]:


business = pd.read_csv('dataset/yelp_academic_dataset_business.csv')


# In[3]:


business.shape


# In[4]:


business.head()


# In[5]:


business = business[pd.notnull(business['categories'])]
business.shape


# In[9]:


indexList = []
for i in range(len(business)):
    if 'Restaurants' not in business['categories'].iloc[i]:
        indexList.append(i)
business=business.drop(business.index[indexList])
business = business[['business_id','categories']]
business


# In[10]:


business.to_csv("onlyRestaurants.csv",index=False)






