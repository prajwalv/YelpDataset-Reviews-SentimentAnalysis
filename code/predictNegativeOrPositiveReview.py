#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


# In[97]:


reviews = pd.read_csv('./dataset/FilteredRestaurantReviews.csv')


# In[98]:


review_class = reviews[(reviews['stars'] == 1) | (reviews['stars'] == 2)| (reviews['stars'] == 3)|(reviews['stars'] == 4)| (reviews['stars'] == 5)]
review_class.shape


# In[99]:


A = review_class['text']
B = review_class['stars']


# In[100]:


# remove stopwords and punctuations
import string
def clean_review(review):
    clean_data = [char for char in review if char not in string.punctuation]
    clean_data = ''.join(clean_data)
    
    return [word for word in clean_data.split() if word.lower() not in stopwords.words('english')]


# In[101]:


# to convert the text documents into a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
bag_of_word_transformer = CountVectorizer(analyzer=clean_review).fit(A)


# In[102]:


A = bag_of_word_transformer.transform(A)


# In[103]:


print('Shape of Sparse Matrix: ', A.shape)
print('Amount of Non-Zero occurrences: ', A.nnz)
# Percentage of non-zero values
density = (100.0 * A.nnz / (A.shape[0] * A.shape[1]))
print("Density: {}".format((density)))


# In[104]:


# Decides whether the review is positive or negative based on the prediciton
def decidePositiveOrNegative(res):
    if res <= 2:
        print("Negative Comment")
    else:
        print("Positive Comment")


# In[105]:


# transforms into document-term matrix 
from sklearn.feature_extraction.text import TfidfTransformer  
tfidfconverter = TfidfTransformer()  
A = tfidfconverter.fit_transform(A).toarray() 


# In[106]:


# splits the dataset into train and test samples, 70% of the dataset 30% of the dataset is the test data.
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.3, random_state=101)


# In[107]:


from sklearn.naive_bayes import MultinomialNB
naiveBayes = MultinomialNB()
naiveBayes.fit(A_train, B_train)


# In[108]:


prediction = naiveBayes.predict(A_test)


# In[109]:


print ("Prediction using Naive Bayes model\n")
for res in prediction:
    decidePositiveOrNegative(res)


# In[110]:


from sklearn.metrics import f1_score
f1_scr  = []
f1_scr.append(f1_score(B_test,prediction,average='micro')*100)


# In[111]:


from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(A_train, B_train)


# In[112]:


prediction = svc.predict(A_test)


# In[113]:


print ("Prediction using Linear SVC model\n")
for res in prediction:
    decidePositiveOrNegative(res)


# In[114]:


f1_scr.append(f1_score(B_test,prediction,average='micro')*100)


# In[115]:


from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(A_train, B_train)


# In[116]:


prediction = LogReg.predict(A_test)


# In[117]:


print ("Prediction using Logistic Regression model\n")
for res in prediction:
    decidePositiveOrNegative(res)


# In[118]:


f1_scr.append(f1_score(B_test,prediction,average='micro')*100)


# In[119]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=18)
rf.fit(A_train, B_train)
prediction = rf.predict(A_test)


# In[120]:


print ("Prediction using Random Forest Classifier model\n")
for res in prediction:
    decidePositiveOrNegative(res)


# In[121]:


f1_scr.append(f1_score(B_test,prediction,average='weighted')*100)


# In[122]:


import matplotlib.pyplot as plt
line1 = plt.plot (
          ['Naive Bayes','Linear SVC','LogisticRegression','RandomForest Classifier'],f1_scr ,'--o',alpha=0.7)
plt.title("Model Evaluation")
plt.ylabel('F1 score')
plt.xlabel('Models')
plt.show()




