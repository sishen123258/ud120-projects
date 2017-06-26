#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import pickle
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
import numpy as np
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf=GaussianNB()
clf.fit(features_train, labels_train)
res=clf.predict(features_test)

length = range(0, len(res))
acc=0
for i in length:
    if(res[i]==labels_test[i]):
        acc=acc+1

print acc/float(len(res))

#########################################################

# dicObj={1:1,2:2,3:3}
# f=open("./pickle.txt","w")
# pickle.dump(dicObj,f)


