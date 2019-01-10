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


business = pd.read_csv('./dataset/onlyRestaurants.csv')
business.head()

reviews = pd.read_csv('./dataset/yelp_academic_dataset_review.csv')


# In[3]:


business.shape


# In[4]:


reviews.shape


# In[5]:


business.head()


# In[6]:


reviews.head(100)


# In[9]:


business_id = business['business_id']
reviews = pd.read_csv("./dataset/yelp_academic_dataset_review.csv",nrows=100000) #considering only a chunk of reviews
reviews = reviews.drop(['useful','funny','cool','date','user_id','review_id'], axis=1)
reviews.head()
rbusiness_id = reviews ['business_id']
print(len(rbusiness_id))


# In[10]:


reviews.head()
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


# In[11]:


reviews.head()


# In[12]:


common = intersection(rbusiness_id, business_id)
print(len(common))


# In[13]:


for i in range (len(common)):
    if reviews.iloc[i]['business_id'] in common:
         reviews.to_csv("./dataset/FilteredRestaurantReviews.csv",index=False)




