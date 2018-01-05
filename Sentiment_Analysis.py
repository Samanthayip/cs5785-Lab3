
# coding: utf-8

# In[196]:

import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
import sklearn
from sklearn import preprocessing
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
from sklearn import datasets, cross_validation, metrics, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.naive_bayes import BernoulliNB

yelp_labels = []
amazon_labels = []
imdb_labels = []


# In[197]:

amazon_content = open('amazon_cells_labelled.txt','r')
yelp_content = open('yelp_labelled.txt','r')
imdb_content = open('imdb_labelled.txt','r')

amazon_content = amazon_content.readlines()
yelp_content = yelp_content.readlines()
imdb_content = imdb_content.readlines()


# In[198]:

amazon_parse = []
yelp_parse = []
imdb_parse = []

for row in amazon_content:
    data, label = row.split('\t')
    label, _ = label.split('\n')
    amazon_parse.append([data,label])
    
for row in yelp_content:
    data, label = row.split('\t')
    label, _ = label.split('\n')
    yelp_parse.append([data,label])

for row in imdb_content:
    data, label = row.split('\t')
    label, _ = label.split('\n')
    imdb_parse.append([data,label])


# In[199]:

adf = pd.DataFrame(data = amazon_parse, columns = ['text','label'])
total = adf.groupby(['label']).size()

ydf = pd.DataFrame(data = yelp_parse, columns = ['text','label'])
total = ydf.groupby(['label']).size()

idf = pd.DataFrame(data = imdb_parse, columns = ['text','label'])
total = idf.groupby(['label']).size()
total


# In[200]:

adf['text'] = adf['text'].str.lower()
ydf['text'] = ydf['text'].str.lower()
idf['text'] = idf['text'].str.lower()


# In[201]:

adf['text'] = adf['text'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
ydf['text'] = ydf['text'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
idf['text'] = idf['text'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))


# In[202]:

stop = stopwords.words('english')


# In[203]:

stop.extend(('1','2','3','4','5','6','7','8','9','10'))


# In[204]:

stop


# In[205]:

adf['text'] = adf['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
ydf['text'] = ydf['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
idf['text'] = idf['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[206]:

ydf['text'][0]


# In[207]:

adf_zero = adf.loc[adf['label'] == '0']
ydf_zero = ydf.loc[ydf['label'] == '0']
idf_zero = idf.loc[idf['label'] == '0']


# In[208]:

train_amazon_zero = adf_zero.iloc[:400,:]
train_yelp_zero = ydf_zero.iloc[:400,:]
train_imdb_zero = idf_zero.iloc[:400,:]


# In[209]:

test_amazon_zero = adf_zero.iloc[400:,:]
test_yelp_zero = ydf_zero.iloc[400:,:]
test_imdb_zero = idf_zero.iloc[400:,:]


# In[210]:

adf_one = adf.loc[adf['label'] == '1']
ydf_one = ydf.loc[ydf['label'] == '1']
idf_one = idf.loc[idf['label'] == '1']
train_amazon_one = adf_one.iloc[:400,:]
train_yelp_one = ydf_one.iloc[:400,:]
train_imdb_one = idf_one.iloc[:400,:]
test_amazon_one = adf_one.iloc[400:,:]
test_yelp_one = ydf_one.iloc[400:,:]
test_imdb_one = idf_one.iloc[400:,:]


# # bag of words

# make one pass through all the reviews in the training set and build a dictionary of unique words

# In[211]:

import collections, re


# In[212]:

#sam is all training reviews, both positive and negative
sam = train_amazon_zero.append(train_amazon_one, ignore_index=True).append(train_yelp_one, ignore_index=True).append(train_yelp_zero, ignore_index=True).append(train_imdb_zero, ignore_index=True).append(train_imdb_one, ignore_index=True)
#anna is all TESTING reviews, both positive and negative
anna = test_amazon_zero.append(test_amazon_one, ignore_index=True).append(test_yelp_one, ignore_index=True).append(test_yelp_zero, ignore_index=True).append(test_imdb_zero, ignore_index=True).append(test_imdb_one, ignore_index=True)


# In[213]:

sam = sam['text']
anna = anna['text']
sam


# In[214]:

bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in sam]
bagsofwords_test = [collections.Counter(re.findall(r'\w+', txt)) for txt in anna]
print(len(bagsofwords))
print(len(bagsofwords_test))
print(bagsofwords[2])
bagsofwords[2]['get']


# In[215]:

sumbags = sum(bagsofwords, collections.Counter())
#sumbags is a dictionary for TRAINING words ONLY, but also has the total count


# In[216]:

len(sumbags)
#size of the dictionary


# In[217]:

sumbags.keys()


# The ith element of a review’s feature vector is the number of
# occurrences of the ith dictionary word in the review

# In[218]:

sumbags = collections.OrderedDict(sumbags)
keys = sumbags.keys()
keys


# In[219]:

row_size = 2400
#row_size = len(train_amazon_zero) + len (train_amazon_one)
feature_matrix_train = np.zeros((row_size, len(sumbags)))


# In[220]:

def feature_vector_maker(reviews, feature_matrix, bag):
    for index, row in enumerate(reviews):
        words = re.sub("[^\w]", " ",  row).split()
        for word in words:
            if word not in sumbags.keys():
                next
            else:
                f = list(sumbags.keys()).index(word)
                #for row and column in feature_matrix, pull count of that word
                feature_matrix[index][f] = bag[index][word]
    return feature_matrix


# In[221]:

reviews_training_feature_vectors = feature_vector_maker(sam, feature_matrix_train, bagsofwords)
reviews_training_feature_vectors


# In[222]:

train_normalized = preprocessing.normalize(reviews_training_feature_vectors, norm="l2", axis=1, copy=True, return_norm=False)


# In[223]:

train_normalized


# In[224]:

feature_matrix_test = np.zeros((600, len(sumbags)))
reviews_testing_feature_vectors = feature_vector_maker(anna, feature_matrix_test, bagsofwords_test)
reviews_testing_feature_vectors


# In[225]:

test_normalized = preprocessing.normalize(reviews_testing_feature_vectors, norm="l2", axis=1, copy=True, return_norm=False)


# In[226]:

test_normalized


# In[227]:

np.sum(reviews_testing_feature_vectors[4])


# # Sentiment prediction. Train a logistic regression model

# Train a logistic regression model (you can use existing packages here) on the training set 

# In[228]:

x_trainf, y_trainf, z_trainf = cross_validation.KFold(len(train_normalized[:,:]), n_folds=3)


# In[229]:

train_indices_x = x_trainf[0]
traindata_x = train_normalized[train_indices_x]
train_indices_y = y_trainf[0]
traindata_y = train_normalized[train_indices_y]
train_indices_z = z_trainf[0]
traindata_z = train_normalized[train_indices_z]


# In[230]:

traindata_x.shape


# In[231]:

train_rev = train_amazon_zero.append(train_amazon_one, ignore_index=True).append(train_yelp_one, ignore_index=True).append(train_yelp_zero, ignore_index=True).append(train_imdb_zero, ignore_index=True).append(train_imdb_one, ignore_index=True)
test_rev = test_amazon_zero.append(test_amazon_one, ignore_index=True).append(test_yelp_one, ignore_index=True).append(test_yelp_zero, ignore_index=True).append(test_imdb_zero, ignore_index=True).append(test_imdb_one, ignore_index=True)
train_review = train_rev['label'].apply(int)
test_review = test_rev['label'].apply(int)


# In[232]:

trainlabels_x = train_review[train_indices_x]
trainlabels_y = train_review[train_indices_y]
trainlabels_z = train_review[train_indices_z]


# In[233]:

trainlabels_z.shape


# In[234]:

LogReg = LogisticRegression()


# In[235]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred1 = LogReg.predict(traindata_x)


# In[236]:

sklearn.metrics.accuracy_score(trainlabels_x, y_pred1, normalize=True, sample_weight=None)


# In[237]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred2 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred2, normalize=True, sample_weight=None)


# In[238]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred3 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred3, normalize=True, sample_weight=None)


# ## test on the testing set

# In[239]:

logreg_fit = LogReg.fit(train_normalized[:,:], train_review)


# In[240]:

test_predictions = LogReg.predict(test_normalized)
sklearn.metrics.accuracy_score(test_review, test_predictions, normalize=True, sample_weight=None)


# In[241]:

confmatrix = metrics.confusion_matrix(test_review, test_predictions)
print(confmatrix)
plt.figure()
plt.imshow(confmatrix, cmap='Greens')
plt.savefig("confusionmatrix.png")


# In[242]:

word_weights = LogReg.coef_[0]
important_words = []


# In[243]:

for i, word in enumerate(keys):
    important_words.append((abs(word_weights[i]),(word)))


# In[244]:

important_words.sort(reverse=True)


# In[245]:

important_words


# ## Report the classification accuracy and confusion matrix. Inspecting the weight vector of the logistic regression, what are the words that play the most important roles in deciding the sentiment of the reviews?

# In[246]:

clf = BernoulliNB()


# In[247]:

clf.fit(traindata_x, trainlabels_x)
k1 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, k1, normalize=True, sample_weight=None)


# In[248]:

clf.fit(traindata_y, trainlabels_y)
k2 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, k2, normalize=True, sample_weight=None)


# In[249]:

clf.fit(traindata_z, trainlabels_z)
k3 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, k3, normalize=True, sample_weight=None)


# In[250]:

bernoulli_predictions = clf.predict(reviews_testing_feature_vectors)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# In[251]:

confmatrix = metrics.confusion_matrix(test_review, bernoulli_predictions)
print(confmatrix)
plt.figure()
plt.imshow(confmatrix, cmap='Greens')
plt.savefig("confusionmatrix.png")


# # NGRAM

# N-gram model. Similar to the bag of words model, but now you build up a dictionary of ngrams,
# which are contiguous sequences of words. For example, “Alice fell down the rabbit
# hole” would then map to the 2-grams sequence: ["Alice fell", "fell down", "down the", "the
# rabbit", "rabbit hole"], and all five of those symbols would be members of the n-gram dictionary

# In[252]:

def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i], input_list[i+1]))
  return bigram_list


# In[253]:

sam2 = []
for x in bagsofwords:
    sentence = list(x.keys())
    sam2.append(find_bigrams(sentence))

anna2 = []
for x in bagsofwords_test:
    sentence = list(x.keys())
    anna2.append(find_bigrams(sentence))


# In[254]:

anna2


# In[255]:

bagsofwords2 = [collections.Counter(txt) for txt in sam2]
bagsofwords_test2 = [collections.Counter(txt) for txt in anna2]


# In[256]:

bagsofwords2


# In[257]:

sumbags = sum(bagsofwords2, collections.Counter())
#sumbags is a dictionary for TRAINING words ONLY, but also has the total count


# In[258]:

len(sumbags)


# In[259]:

sumbags = collections.OrderedDict(sumbags)
keys = sumbags.keys()
keys


# In[260]:

feature_matrix_train2 = np.zeros((row_size, len(sumbags)))
feature_matrix_test2 = np.zeros((600, len(sumbags)))


# In[261]:

def feature_vector_maker2(reviews, feature_matrix, bag):
    for index, row in enumerate(reviews):
        for word in row:
            if word not in sumbags.keys():
                next
            else:
                f = list(sumbags.keys()).index(word)
                #for row and column in feature_matrix, pull count of that word
                feature_matrix[index][f] = bag[index][word]
    return feature_matrix


# In[262]:

reviews_training_feature_vectors2 = feature_vector_maker2(sam2, feature_matrix_train2, bagsofwords2)
reviews_training_feature_vectors2


# In[263]:

reviews_testing_feature_vectors2 = feature_vector_maker2(anna2, feature_matrix_test2, bagsofwords_test2)
reviews_testing_feature_vectors2


# In[264]:

np.sum(reviews_training_feature_vectors2[5])


# In[265]:

sam2[5]


# In[266]:

np.sum(reviews_testing_feature_vectors2[7])


# In[267]:

anna2[7]


# ## sentiment prediction w/ n-gram

# In[268]:

x_trainf, y_trainf, z_trainf = cross_validation.KFold(len(reviews_training_feature_vectors2[:,:]), n_folds=3)


# In[269]:

train_indices_x = x_trainf[0]
traindata_x = feature_matrix_train2[train_indices_x]
train_indices_y = y_trainf[0]
traindata_y = feature_matrix_train2[train_indices_y]
train_indices_z = z_trainf[0]
traindata_z = feature_matrix_train2[train_indices_z]


# In[270]:

traindata_x.shape


# In[271]:

trainlabels_x = train_review[train_indices_x]
trainlabels_y = train_review[train_indices_y]
trainlabels_z = train_review[train_indices_z]


# In[272]:

trainlabels_x.shape


# In[273]:

LogReg = LogisticRegression()


# In[274]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred4 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred4, normalize=True, sample_weight=None)


# In[275]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred5 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred5, normalize=True, sample_weight=None)


# In[276]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred6 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred6, normalize=True, sample_weight=None)


# In[277]:

logreg_fit = LogReg.fit(reviews_training_feature_vectors2[:,:], train_review)


# In[278]:

test_predictions = LogReg.predict(reviews_testing_feature_vectors2)
sklearn.metrics.accuracy_score(test_review, test_predictions, normalize=True, sample_weight=None)


# In[279]:

confmatrix = metrics.confusion_matrix(test_review, test_predictions)
print(confmatrix)
plt.figure()
plt.imshow(confmatrix, cmap='Greens')
plt.savefig("confusionmatrix.png")


# In[280]:

clf = BernoulliNB()


# In[281]:

clf.fit(traindata_x, trainlabels_x)
k4 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, k4, normalize=True, sample_weight=None)


# In[282]:

clf.fit(traindata_y, trainlabels_y)
k5 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, k5, normalize=True, sample_weight=None)


# In[283]:

clf.fit(traindata_z, trainlabels_z)
k6 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, k6, normalize=True, sample_weight=None)


# In[284]:

bernoulli_predictions = clf.predict(reviews_testing_feature_vectors2)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# In[285]:

confmatrix = metrics.confusion_matrix(test_review, bernoulli_predictions)
print(confmatrix)
plt.figure()
plt.imshow(confmatrix, cmap='Greens')
plt.savefig("confusionmatrix.png")


# # PCA

# Implement PCA to reduce the dimension of features calculated in (e) to 10, 50 and 100 respectively

# In[286]:

def generatePCA(data, n):
    data = data - np.mean(data)
    U, s, V = np.linalg.svd(data, full_matrices=False)
    V = V.transpose()
    V = V[:, :n]
    return np.dot(data, V)


# In[287]:

pca_train_data_10 = generatePCA(train_normalized, 10)


# In[288]:

pca_train_data_10.shape


# In[289]:

pca_test_data_10 = generatePCA(test_normalized, 10)


# In[290]:

pca_train_data_50 = generatePCA(train_normalized, 50)


# In[291]:

pca_test_data_50 = generatePCA(test_normalized, 50)


# In[292]:

pca_train_data_100 = generatePCA(train_normalized, 100)


# In[293]:

pca_test_data_100 = generatePCA(test_normalized, 100)


# ### Logistic Regression

# with 10

# In[294]:

LogReg = LogisticRegression()


# In[295]:

traindata_x = pca_train_data_10[train_indices_x]
traindata_y = pca_train_data_10[train_indices_y]
traindata_z = pca_train_data_10[train_indices_z]


# In[296]:

traindata_x.shape


# In[297]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred7 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred7, normalize=True, sample_weight=None)


# In[298]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred8 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred8, normalize=True, sample_weight=None)


# In[299]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred9 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred9, normalize=True, sample_weight=None)


# fit ALL training data

# In[300]:

logreg_fit = LogReg.fit(pca_train_data_10, train_review)


# In[301]:

test_predictions_PCA = LogReg.predict(pca_test_data_10)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# with 50

# In[302]:

LogReg = LogisticRegression()


# In[303]:

traindata_x = pca_train_data_50[train_indices_x]
traindata_y = pca_train_data_50[train_indices_y]
traindata_z = pca_train_data_50[train_indices_z]


# In[304]:

traindata_x.shape


# In[305]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred10 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred10, normalize=True, sample_weight=None)


# In[306]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred11 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred11, normalize=True, sample_weight=None)


# In[307]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred12 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred12, normalize=True, sample_weight=None)


# In[308]:

logreg_fit = LogReg.fit(pca_train_data_50, train_review)


# In[309]:

test_predictions_PCA = LogReg.predict(pca_test_data_50)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# with 100

# In[310]:

LogReg = LogisticRegression()


# In[311]:

traindata_x = pca_train_data_100[train_indices_x]
traindata_y = pca_train_data_100[train_indices_y]
traindata_z = pca_train_data_100[train_indices_z]


# In[312]:

traindata_x.shape


# In[313]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred13 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred13, normalize=True, sample_weight=None)


# In[314]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred14 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred14, normalize=True, sample_weight=None)


# In[315]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred15 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred15, normalize=True, sample_weight=None)


# In[316]:

logreg_fit = LogReg.fit(pca_train_data_100, train_review)


# In[317]:

test_predictions_PCA = LogReg.predict(pca_test_data_100)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# In[318]:

confmatrix = metrics.confusion_matrix(test_review, test_predictions_PCA)
print(confmatrix)


# ### Bernoulli PCA

# In[319]:

clf = BernoulliNB()


# In[320]:

traindata_x = pca_train_data_10[train_indices_x]
traindata_y = pca_train_data_10[train_indices_y]
traindata_z = pca_train_data_10[train_indices_z]


# In[321]:

clf.fit(traindata_x, trainlabels_x)
pca_10_bn1 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, pca_10_bn1, normalize=True, sample_weight=None)


# In[322]:

clf.fit(traindata_y, trainlabels_y)
pca_10_bn2 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, pca_10_bn2, normalize=True, sample_weight=None)


# In[323]:

clf.fit(traindata_z, trainlabels_z)
pca_10_bn3 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, pca_10_bn3, normalize=True, sample_weight=None)


# In[324]:

traindata_x.shape


# In[325]:

bernoulli_predictions = clf.predict(pca_test_data_10)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# with 50

# In[326]:

clf = BernoulliNB()


# In[327]:

traindata_x = pca_train_data_50[train_indices_x]
traindata_y = pca_train_data_50[train_indices_y]
traindata_z = pca_train_data_50[train_indices_z]


# In[328]:

clf.fit(traindata_x, trainlabels_x)
pca_50_bn1 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, pca_50_bn1, normalize=True, sample_weight=None)


# In[329]:

clf.fit(traindata_y, trainlabels_y)
pca_50_bn1 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, pca_50_bn1, normalize=True, sample_weight=None)


# In[330]:

clf.fit(traindata_z, trainlabels_z)
pca_50_bn3 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, pca_50_bn3, normalize=True, sample_weight=None)


# In[331]:

bernoulli_predictions = clf.predict(pca_test_data_50)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# with 100

# In[332]:

traindata_x = pca_train_data_100[train_indices_x]
traindata_y = pca_train_data_100[train_indices_y]
traindata_z = pca_train_data_100[train_indices_z]


# In[333]:

clf.fit(traindata_x, trainlabels_x)
pca_100_bn1 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, pca_100_bn1, normalize=True, sample_weight=None)


# In[334]:

clf.fit(traindata_y, trainlabels_y)
pca_100_bn1 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, pca_100_bn1, normalize=True, sample_weight=None)


# In[335]:

clf.fit(traindata_z, trainlabels_z)
pca_100_bn1 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, pca_100_bn1, normalize=True, sample_weight=None)


# In[336]:

bernoulli_predictions = clf.predict(pca_test_data_100)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# ### PCA with 2-gram

# In[337]:

pca_train_data_10 = generatePCA(reviews_training_feature_vectors2, 10)


# In[338]:

pca_test_data_10 = generatePCA(reviews_testing_feature_vectors2, 10)


# In[339]:

pca_train_data_50 = generatePCA(reviews_training_feature_vectors2, 50)


# In[340]:

pca_test_data_50 = generatePCA(reviews_testing_feature_vectors2, 50)


# In[341]:

pca_train_data_100 = generatePCA(reviews_training_feature_vectors2, 100)


# In[342]:

pca_test_data_100 = generatePCA(reviews_testing_feature_vectors2, 100)


# #### PCA, 2-gram, Logistic Regression

# with 10

# In[343]:

LogReg = LogisticRegression()


# In[344]:

x_trainf, y_trainf, z_trainf = cross_validation.KFold(len(reviews_training_feature_vectors2[:,:]), n_folds=3)


# In[345]:

train_indices_x = x_trainf[0]
traindata_x = pca_train_data_10[train_indices_x]
train_indices_y = y_trainf[0]
traindata_y = pca_train_data_10[train_indices_y]
train_indices_z = z_trainf[0]
traindata_z = pca_train_data_10[train_indices_z]


# In[346]:

traindata_x.shape


# In[347]:

# trainlabels_x = train_review[train_indices_x]
# trainlabels_y = train_review[train_indices_y]
# trainlabels_z = train_review[train_indices_z]


# In[348]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred20 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred20, normalize=True, sample_weight=None)


# In[349]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred30 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred30, normalize=True, sample_weight=None)


# In[350]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred40 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred40, normalize=True, sample_weight=None)


# In[351]:

logreg_fit = LogReg.fit(pca_train_data_10, train_review)


# In[352]:

test_predictions_PCA = LogReg.predict(pca_test_data_10)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# with 50

# In[353]:

LogReg = LogisticRegression()


# In[354]:

train_indices_x = x_trainf[0]
traindata_x = pca_train_data_50[train_indices_x]
train_indices_y = y_trainf[0]
traindata_y = pca_train_data_50[train_indices_y]
train_indices_z = z_trainf[0]
traindata_z = pca_train_data_50[train_indices_z]


# In[355]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred50 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred50, normalize=True, sample_weight=None)


# In[356]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred60 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred60, normalize=True, sample_weight=None)


# In[357]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred70 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred70, normalize=True, sample_weight=None)


# In[358]:

test_predictions_PCA = LogReg.predict(pca_test_data_50)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# with 100

# In[359]:

LogReg = LogisticRegression()


# In[360]:

train_indices_x = x_trainf[0]
traindata_x = pca_train_data_100[train_indices_x]
train_indices_y = y_trainf[0]
traindata_y = pca_train_data_100[train_indices_y]
train_indices_z = z_trainf[0]
traindata_z = pca_train_data_100[train_indices_z]


# In[361]:

logreg_fit = LogReg.fit(traindata_x, trainlabels_x)
y_pred200 = LogReg.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred200, normalize=True, sample_weight=None)


# In[362]:

logreg_fit = LogReg.fit(traindata_y, trainlabels_y)
y_pred300 = LogReg.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred300, normalize=True, sample_weight=None)


# In[363]:

logreg_fit = LogReg.fit(traindata_z, trainlabels_z)
y_pred400 = LogReg.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred400, normalize=True, sample_weight=None)


# In[364]:

test_predictions_PCA = LogReg.predict(pca_test_data_100)
sklearn.metrics.accuracy_score(test_review, test_predictions_PCA, normalize=True, sample_weight=None)


# #### PCA, 2-gram, Binomail Gaussian

# In[365]:

clf = BernoulliNB()


# In[366]:

traindata_x = pca_train_data_10[train_indices_x]
traindata_y = pca_train_data_10[train_indices_y]
traindata_z = pca_train_data_10[train_indices_z]


# In[367]:

clf_fit = clf.fit(traindata_x, trainlabels_x)
y_pred200 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, y_pred200, normalize=True, sample_weight=None)


# In[368]:

clf_fit = clf.fit(traindata_y, trainlabels_y)
y_pred300 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, y_pred300, normalize=True, sample_weight=None)


# In[369]:

clf_fit = clf.fit(traindata_z, trainlabels_z)
y_pred400 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, y_pred400, normalize=True, sample_weight=None)


# In[370]:

bernoulli_predictions = clf.predict(pca_test_data_10)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# #### with 50

# In[371]:

clf = BernoulliNB()


# In[372]:

traindata_x = pca_train_data_50[train_indices_x]
traindata_y = pca_train_data_50[train_indices_y]
traindata_z = pca_train_data_50[train_indices_z]


# In[373]:

clf.fit(traindata_x, trainlabels_x)
pca_50_bn2 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, pca_50_bn2, normalize=True, sample_weight=None)


# In[374]:

clf.fit(traindata_y, trainlabels_y)
pca_50_bn22 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, pca_50_bn22, normalize=True, sample_weight=None)


# In[375]:

clf.fit(traindata_z, trainlabels_z)
pca_50_bn23 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, pca_50_bn23, normalize=True, sample_weight=None)


# In[376]:

bernoulli_predictions = clf.predict(pca_test_data_50)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# #### with 100

# In[377]:

clf = BernoulliNB()


# In[378]:

traindata_x = pca_train_data_100[train_indices_x]
traindata_y = pca_train_data_100[train_indices_y]
traindata_z = pca_train_data_100[train_indices_z]


# In[379]:

clf.fit(traindata_x, trainlabels_x)
pca_50_bn2 = clf.predict(traindata_x)
sklearn.metrics.accuracy_score(trainlabels_x, pca_50_bn2, normalize=True, sample_weight=None)


# In[380]:

clf.fit(traindata_y, trainlabels_y)
pca_50_bn22 = clf.predict(traindata_y)
sklearn.metrics.accuracy_score(trainlabels_y, pca_50_bn22, normalize=True, sample_weight=None)


# In[381]:

clf.fit(traindata_z, trainlabels_z)
pca_50_bn23 = clf.predict(traindata_z)
sklearn.metrics.accuracy_score(trainlabels_z, pca_50_bn23, normalize=True, sample_weight=None)


# In[382]:

logreg_fit = LogReg.fit(pca_train_data_100, train_review)


# In[383]:

bernoulli_predictions = clf.predict(pca_test_data_100)
sklearn.metrics.accuracy_score(test_review, bernoulli_predictions, normalize=True, sample_weight=None)


# In[ ]:



