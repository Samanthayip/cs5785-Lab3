
# coding: utf-8

# In[318]:

import numpy as np
import math
from scipy import misc
from scipy.stats.distributions import norm
from matplotlib import pylab as plt
import matplotlib.cm as cm
import sklearn
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import csv
import pandas as pd
get_ipython().magic('matplotlib inline')
import string
import nltk
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn import datasets, cross_validation, metrics, model_selection
from scipy.spatial import distance
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.naive_bayes import BernoulliNB
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from scipy.spatial import distance


# In[319]:

ofdata = []
i = 1
for line in open('./oldf.txt'):
    if i < 100:
        ofdata.append(line.strip("\n").split('      '))
    else:
        ofdata.append(line.strip("\n").split('     '))
    i = i + 1
for x in ofdata:
    x[1] = x[1].strip()


# In[320]:

ofdata = np.array(ofdata, dtype=np.float32)


# In[321]:

for x in ofdata:
    x[1] = float(x[1])
    x[2] = float(x[2])


# Parse and plot all data points on 2-D plane.

# In[322]:

a = [el[1] for el in ofdata]
b = [el[2] for el in ofdata]


# In[323]:

plt.scatter(a, b)
plt.xlabel('erruptions')
plt.ylabel('waiting')


# In[324]:

ofdata = ofdata[:,1:]


# In[325]:

a, b = np.random.choice(ofdata.shape[0],size = 2)
print(a)
print(b)


# In[ ]:




# In[326]:

#pick mu in a better way, potential to simplify for sigma and pi
def initparams():
    #based on graph above
    a, b = np.random.choice(ofdata.shape[0],size = 2)
    mu_1 = ofdata[a]
    mu_2 = ofdata[b]
    sigma_squared_1 = np.identity(2)
    sigma_squared_2 = np.identity(2)
    mus = [mu_1, mu_2]
    sigma_squares = [sigma_squared_1, sigma_squared_2]
    pi = 0.5
    pi2 = 1 - pi
#     pi = random.uniform(0.01,.99)
    return mu_1,mu_2,sigma_squared_1,sigma_squared_2,pi, pi2


# In[327]:

def ExpStep(x, mu1, mu2, sigma1, sigma2, pi, pi2):
    matrix = np.zeros((272,2))
    for index, i in enumerate(x):
        #make sure sigma1 and 2 are square matrices
        matrix[index,0] = pi * multivariate_normal.pdf(i, mu1, sigma1)
        matrix[index,1] = pi2 * multivariate_normal.pdf(i, mu2, sigma2)   
    #equal to the denominator
    responsibilities = preprocessing.normalize(matrix, norm="l1", axis=1, copy=True, return_norm=False)

    return responsibilities


# In[328]:

#x is the whole array, thus the function returns the whole array
def MaxStep(x, responsibilities):
    mu_1 = np.average(x, axis=0, weights=responsibilities[:,0], returned=False)
    mu_2 = np.average(x, axis=0, weights=responsibilities[:,1], returned=False)

    distance_matrix = distance.cdist(x, [mu_1, mu_2], metric='euclidean', p=None, V=None, VI=None, w=None)
    sigma_1 = distance_matrix[:,0]
    sigma_2 = distance_matrix[:,1]
    sigma_squared_1 = np.average(np.square(sigma_1),axis=0, weights=responsibilities[:,0], returned=False)
    sigma_squared_2 = np.average(np.square(sigma_2),axis=0, weights=responsibilities[:,1], returned=False)
    covariance_1 = sigma_squared_1 * np.identity(2)
    covariance_2 = sigma_squared_2 * np.identity(2)
    
    pi = np.mean(responsibilities[:,0])
    pi2 = np.mean(responsibilities[:,1])
        
    return mu_1, mu_2 , covariance_1, covariance_2, pi, pi2


# In[329]:

def notconverge(cluster_means):
    #if length of the list of cluster means is equal or less than 1, then return True
    if (len(cluster_means) <= 1):
        return True
    new_point = cluster_means[-1]
    old_point = cluster_means[-2]
    distance_points = (distance.euclidean(new_point,old_point))
    return (distance_points >= 0.0001)


# In[342]:

cluster_mean1 = []
cluster_mean2 = []
mu1,mu2,sigma1,sigma2,pi, pi2= initparams()
cluster_mean1.append(mu1)
cluster_mean2.append(mu2)
num_iterations = 0

params=[]
not_converged = True
while not_converged and num_iterations <= 100:
    responsibilities = ExpStep(ofdata, mu1, mu2, sigma1, sigma2, pi, pi2)
    mu1,mu2,sigma1,sigma2,pi, pi2 = MaxStep(ofdata,responsibilities)
    cluster_mean1.append(mu1)
    cluster_mean2.append(mu2)
    not_converged = notconverge(cluster_mean1) or notconverge(cluster_mean2)
    num_iterations += 1
    print(num_iterations)
print("DONEEE!!!!!")


# In[343]:

a = []
b = []
for x in cluster_mean1:
    a.append(x[0])
    b.append(x[1])


# In[344]:

c = []
d = []
for x in cluster_mean2:
    c.append(x[0])
    d.append(x[1])


# - Plot the trajectories of two mean vectors in 2 dimensions (i.e., coordinates vs. iteration)

# In[345]:

eruptions = ofdata[:,0]
waiting = ofdata[:,1]
plt.plot(eruptions, waiting, '.')
plt.plot(a,b, "ro", label="mu_1", linestyle='--')
plt.plot(c,d, "go", label="mu_2", linestyle='--')
plt.xlabel("eruptions")
plt.ylabel("waiting")
plt.title("Movement of means")
plt.legend()
plt.show()


# 
# 
# - Run your program for 50 times with different initial parameter guesses. Show the distribution of the total number of iterations needed for algorithm to converge.

# In[334]:

ind_num_iterations = []
iteration_number = []
for i in range(50):
    cluster_mean1 = []
    cluster_mean2 = []
    mu1,mu2,sigma1,sigma2,pi, pi2= initparams()
    cluster_mean1.append(mu1)
    cluster_mean2.append(mu2)
    num_iterations = 0

    params=[]
    not_converged = True
    while not_converged and num_iterations <= 100:
        responsibilities = ExpStep(ofdata, mu1, mu2, sigma1, sigma2, pi, pi2)
        mu1,mu2,sigma1,sigma2,pi, pi2 = MaxStep(ofdata,responsibilities)
        cluster_mean1.append(mu1)
        cluster_mean2.append(mu2)
        not_converged = notconverge(cluster_mean1) or notconverge(cluster_mean2)
        num_iterations += 1
#         print(num_iterations)
    ind_num_iterations.append(num_iterations)
    print(i)
    print("DONEEE!!!!!")


# In[335]:

ind_num_iterations


# In[336]:

plt.hist(ind_num_iterations)
plt.title("Distribution of iterations")
plt.show()


# In[337]:

from sklearn.cluster import KMeans


num_iterations = 0
kmeans = KMeans(n_clusters=2, random_state=0).fit(ofdata) 
        
cluster_mean1 = []
cluster_mean2 = []

mu1,mu2,sigma1,sigma2,pi, pi2= initparams()
mu1 = kmeans.cluster_centers_[0]
mu2 = kmeans.cluster_centers_[1]


cluster_mean1.append(mu1)
cluster_mean2.append(mu2)

not_converged = True

while not_converged and num_iterations <= 100:
    responsibilities = ExpStep(ofdata, mu1, mu2, sigma1, sigma2, pi, pi2)
    mu1,mu2,sigma1,sigma2,pi, pi2 = MaxStep(ofdata,responsibilities)
    cluster_mean1.append(mu1)
    cluster_mean2.append(mu2)
    not_converged = notconverge(cluster_mean1) or notconverge(cluster_mean2)
    num_iterations += 1
    print(num_iterations)
print("DONEEE!!!!!")
   


# In[338]:

cluster_mean1


# In[339]:

from matplotlib import pyplot as plt

eruptions_1 = np.array(cluster_mean1)[:,0]
waiting_1 = np.array(cluster_mean1)[:,1]

eruptions_2 = np.array(cluster_mean2)[:,0]
waiting_2 = np.array(cluster_mean2)[:,1]

plt.xlabel('erruptions')
plt.ylabel('waiting')

eruptions = ofdata[:,0]
waiting = ofdata[:,1]

plt.plot(eruptions, waiting, '.')
plt.plot(eruptions_1, waiting_1, "ro", linestyle='--', label = "mu_1")
plt.plot(eruptions_2, waiting_2, "go", linestyle='--', label = "mu_2")
plt.show()


# In[340]:

from sklearn.cluster import KMeans
n=0
all_iterations = []
ind_num_iterations = []

for i in range(50):
    num_iterations = 0
    kmeans = KMeans(n_clusters=2, random_state=0).fit(ofdata) 

    cluster_mean1 = []
    cluster_mean2 = []

    mu1,mu2,sigma1,sigma2,pi, pi2= initparams()
    mu1 = kmeans.cluster_centers_[0]
    mu2 = kmeans.cluster_centers_[1]


    cluster_mean1.append(mu1)
    cluster_mean2.append(mu2)

    not_converged = True

    while not_converged and num_iterations <= 100:
        responsibilities = ExpStep(ofdata, mu1, mu2, sigma1, sigma2, pi, pi2)
        mu1,mu2,sigma1,sigma2,pi, pi2 = MaxStep(ofdata,responsibilities)
        cluster_mean1.append(mu1)
        cluster_mean2.append(mu2)
        not_converged = notconverge(cluster_mean1) or notconverge(cluster_mean2)
        num_iterations += 1
        print(num_iterations)
    ind_num_iterations.append(num_iterations)
    print("DONEEE!!!!!")


# In[341]:

plt.hist(ind_num_iterations)
plt.xlabel('iterations')
plt.ylabel('count')
plt.show()

