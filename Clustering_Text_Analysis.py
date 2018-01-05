
# coding: utf-8

# # 2a: Algorithm for getting top 10 titles

# In[164]:

import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as c
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from collections import *
from scipy.spatial import distance
import nltk as nl
import csv
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[183]:

titles_text = open("science2k-titles.txt")
titles = []


# In[184]:

for row in titles_text:
    titles.append(row)
titles


# In[185]:

document = np.load("science2k-doc-word.npy")


# In[186]:

document


# In[187]:

no_clusters = 5
model_2 = KMeans(n_clusters= no_clusters, init='k-means++', max_iter=100, n_init=10)
model_2.fit(document)


# In[188]:

labels = model_2.labels_
labels


# In[189]:

centers = model_2.cluster_centers_
centers


# In[190]:

names = ['Cluster'+str(i) for i in range(no_clusters)]
dict1 = {name :{} for name in names}

for index, i in enumerate(labels):
    name = 'Cluster'+str(i)
    dict1[name][index] = [distance.euclidean(document[index],centers[i])]
print(dict1)


# In[191]:

top_10_titles = [[] for i in range(no_clusters)]
for i in range(no_clusters):
    list = []
    temp = []
    name = 'Cluster'+str(i)
#     list = sorted(dict1[name], key=dict1[name].get, reverse=True)[:10]
    list = sorted(dict1[name], key=dict1[name].get)[:10]
    print(list)
    for index in list:
        temp.append(titles[index])
    top_10_titles[i].append(temp)
top_10_titles


# In[192]:

for index, i in enumerate(top_10_titles):
    print ('Cluster', str(index))
    print(i)


# # 2b: Algorithm for getting top 10 vocabs

# In[174]:

v_text = open("science2k-vocab.txt")
vocabs = []
for row in v_text:
    vocabs.append(row)
vocabs


# In[194]:

v_document = np.load("science2k-word-doc.npy")
v_document
v_document.shape


# In[200]:

no_v_clusters = 6
v_model = KMeans(n_clusters= no_v_clusters, init='k-means++', max_iter=100, n_init=10)
v_model.fit(v_document)


# In[201]:

v_labels = v_model.labels_
v_labels


# In[202]:

v_centers = v_model.cluster_centers_
v_centers


# In[203]:

names = ['Cluster'+str(i) for i in range(no_v_clusters)]
dict = {name :{} for name in names}
for index, i in enumerate(v_labels):
    name = 'Cluster'+str(i)
    dict[name][index] = [distance.euclidean(v_document[index],v_centers[i])]
print(dict)


# In[204]:

top_10_document = [[] for i in range(no_v_clusters)]
for i in range(no_v_clusters):
    list = []
    temp = []
    name = 'Cluster'+str(i)
    list = sorted(dict[name], key=dict[name].get)[:10]
    print(list)
    for index in list:
        temp.append(vocabs[index])
    top_10_document[i].append(temp)
top_10_document


# In[205]:

for index, i in enumerate(top_10_document):
    print ('Cluster', str(index))
    print(i)


# # 2a: Finding the optimal k

# In[18]:

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt 

# k means determine k
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(document)
    kmeanModel.fit(document)
    distortions.append(sum(np.min(cdist(document,            kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / document.shape[0])
 

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# # 2b: Finding the optimal k

# In[19]:

# k means determine k
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(v_document)
    kmeanModel.fit(v_document)
    distortions.append(sum(np.min(cdist(v_document,        kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / v_document.shape[0])
 

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# 
