#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import collections, numpy
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
classifier = svm.SVC(kernel='rbf', C=10000.)#try several values of C (say, 10.0, 100., 1000., and 10000.)
t0 = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
labels_predicted = classifier.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
array = numpy.array(labels_predicted)
count = (array == 1).sum()
print count
accuracyScore = accuracy_score(labels_test, labels_predicted)
print accuracyScore
print "\n"
#########################################################
#0 or 1, corresponding to Sara and Chris respectively

