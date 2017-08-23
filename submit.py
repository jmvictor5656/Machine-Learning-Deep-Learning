#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:08:59 2017

@author: kunal
"""

import pandas as pd
import numpy as np
import nltk.data
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

document_set = pd.read_csv('document_set.csv')

training_data = pd.read_csv('Training_Data.csv')

data=training_data.merge(document_set,how='left',left_on='document_id',right_on='Document_Id')

data=data[['Text','category']]

text=np.array(data['Text'])

label=np.array(data['category'])

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def text_processing(doc):
    """
    function for creating an array of words according to category and removing
    all punctuation marks, stop words
    
    doc:array of sentences 
    ---- 
    
    return:2D array of words and location of category index equals to index of
            Text for same category
    ------
    """
    doc=[re.sub("[^a-zA-Z]"," ", s) for s in doc]
    doc=[s.lower().split() for s in doc]
    stops = set(stopwords.words("english"))
    doc=[[doc[i][j] for j in range(len(doc[i]))if not doc[i][j] in stops]for i in range(len(doc)) ]
    return doc

text=text_processing(text)

test_df = pd.read_csv('Test_Data.csv')

test_set=test_df.merge(document_set,how='left',left_on='document_id',right_on='Document_Id')

test_set=test_set[['document_id','Text']]

test_text=text_processing(np.array(test_set['Text']))

size=300 #size of vector
min_count=5 #minm count for length of sentence
workers=4 #for parallel processing
window=10
sample = 1e-3 #for downsampling

from gensim.models import word2vec
model = word2vec.Word2Vec(text, workers=workers, \
            size=size, min_count = min_count, \
            window = window, sample = sample)
model.save('model')


def createFeature(doc, vectorModel, size):
    """
    for creating vector of words
    
    Parameter:doc , it is 1D array of words representing a sentence
    ---------vectormodel:it is the model we designed above using word2vec
              size:size of vector
    
    Returns: a 1D array of vector
    
    
    """
    featureToVec = np.zeros((size,),dtype="float32")
    no_of_words=0.
    index2word = set(model.wv.index2word)
    
    for word in doc:
        if word in index2word:
            no_of_words+=1
            featureToVec = np.add(featureToVec,vectorModel[word])
    featureToVec = np.divide(featureToVec,no_of_words)
    return featureToVec

def getFeatureVector(doc, vectorModel, size):
    """
    for creating vector of complete text document i.e a 2D array of words
    
    Parameter:doc, 2D array of words
    --------- vectorModel, model we have define using word2vec
              size:int size of each vector
    Returns: Returns a 2D array of vectors representing words         
    --------
    """
    count=0
    featureToVecs=np.zeros((len(doc), size),dtype='float32')
    for d in doc:
        if count%2000==0:
            print('Text %d of %d done'%(count,len(doc)))
        featureToVecs[count]=createFeature(d,model,size)
        count +=1
    return featureToVecs

trainDataVecs = getFeatureVector( text, model, size )  #vectors for train data
testDataVecs=   getFeatureVector( test_text, model, size )#vectors for testing

clf = RandomForestClassifier( n_estimators = 100 )#RandomForestClassifier

trainDataVecs = Imputer().fit_transform(trainDataVecs)#for removing out of range value
testDataVecs=Imputer().fit_transform(testDataVecs) 

clf=clf.fit(trainDataVecs,label)

predict=clf.predict(testDataVecs)

submit=pd.DataFrame(predict,columns=['category'])

test_df_c=pd.DataFrame(test_df['document_id'])

result=test_df_c.join(submit,how='outer')

result.to_csv('result.csv')#result file

