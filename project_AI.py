# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 23:52:02 2016

@author: jyothsna
"""
import random as rn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas   
#import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import time

from random import shuffle
import csv



global train_data
global train_class
global test_data
global test_class

def f1score(y_true,y_pred):
    f1=f1_score(y_true, y_pred, average='macro')
    return f1
def precision(y_true,y_pred):
    pre=precision_score(y_true, y_pred, average='macro')
    return pre
def recall(y_true,y_pred):
    recall=recall_score(y_true, y_pred, average='macro')
    return recall
#knn cannot be done with boosting
def Boosting():
    classifiers=[#MultinomialNB(),
                 #svm.SVC(kernel="poly", C=1),
                #linear_model.LogisticRegression(penalty='l2',C=250),
                #RandomForestClassifier(n_estimators=55),
                #tree.DecisionTreeClassifier(),
                SGDClassifier(loss="hinge", penalty="l2"),
                Perceptron(penalty='l2', alpha=0.00001, fit_intercept=True)
                ]
    classifier_names=['sgd','perceptron']#['NaiveBayes','SVM','LogisticRegression','RandomForest','DecisionTree']
    i=0
    for classifier in classifiers:
        clf=AdaBoostClassifier(base_estimator=classifier, n_estimators=100,algorithm='SAMME')
        start = time.time()
        clf.fit(train_data,train_class)
        
        prediction = clf.predict(test_data)
        end = time.time()
        #scores = cross_val_score(clf, train_data, train_class)
        print('Adaboost '+classifier_names[i]+' F1 %f,Recall %f, Precision %f,time %f'%( f1score(test_class,prediction) , precision(test_class,prediction) , recall(test_class,prediction), end-start ))
        i=i+1
        
def Bagging():
    classifiers=[#MultinomialNB(),
                 #svm.SVC(kernel="poly", C=1),
                #linear_model.LogisticRegression(penalty='l2',C=250),
                #RandomForestClassifier(n_estimators=55),
                #tree.DecisionTreeClassifier(),
                #KNeighborsClassifier(n_neighbors=6),
                SGDClassifier(loss="hinge", penalty="l2"),
                Perceptron(penalty='l2', alpha=0.00001, fit_intercept=True)]
    classifier_names=['sgd','perceptron']#['NaiveBayes','SVM','LogisticRegression','RandomForest','DecisionTree','knn']
    i=0
    for classifier in classifiers:
        clf=BaggingClassifier(classifier, n_estimators=100)
        start = time.time() 
        clf.fit(train_data,train_class)
        prediction = clf.predict(test_data)
        end=time.time()
        print('Bagging '+classifier_names[i]+' F1 %f,Recall %f, Precision %f,time %f'%( f1score(test_class,prediction) , precision(test_class,prediction) , recall(test_class,prediction), end-start ))
        i=i+1
        
def classification():
    classifiers=[#MultinomialNB(),
                 #svm.SVC(kernel="poly", C=1),
                #linear_model.LogisticRegression(penalty='l2',C=250),
                #RandomForestClassifier(n_estimators=55),
                #tree.DecisionTreeClassifier(),
                #KNeighborsClassifier(n_neighbors=6),
                SGDClassifier(loss="hinge", penalty="l2"),
                Perceptron(penalty='l2', alpha=0.00001, fit_intercept=True)]
    classifier_names=['sgd','perceptron']#['NaiveBayes','SVM','LogisticRegression','RandomForest','DecisionTree','knn','sgd','perceptron']
    i=0
    for classifier in classifiers:
        clf=classifier
        start = time.time()
        clf.fit(train_data,train_class)
        end = time.time()
        
        prediction = clf.predict(test_data)
        #scores = cross_val_score(clf, train_data, train_class)
        print(classifier_names[i]+' F1 %f,Recall %f, Precision %f,time %f'%( f1score(test_class,prediction) , precision(test_class,prediction) , recall(test_class,prediction), end-start ))
        i=i+1

def LoadData1(path):
    global train_data
    global train_class
    global test_data
    global test_class

    data = pandas.read_csv(path, sep=",", header = None)
    data=data.replace(['vhigh', 'high', 'med', 'low',
                  '2', '3', '4', '5more','more',
                  'small', 'big'], 
                     [4,3,2,1,2,3,4,6,6,1,3]) 
    #print(data)
    data=data.apply(preprocessing.LabelEncoder().fit_transform)
    data=data.values.tolist()
    rn.shuffle(data)
    
    #deviding the data in to test set and training set
    size=int(len(data)*0.9)
    train_data=data[0:size]
    test_data=data[size:len(data)]
        
    train_class=np.array(train_data)[:,len(train_data[0])-1]
    train_data=np.array(train_data)[:,range(0,len(train_data[0])-1)]

    test_class=np.array(test_data)[:,len(test_data[0])-1]
    test_data=np.array(test_data)[:,range(0,len(test_data[0])-1)]
    
    
def LoadData1(path):
    global train_data
    global train_class
    global test_data
    global test_class
    data = []
    with open(path, 'rb') as csvfile:
        data1 = csv.reader(csvfile, delimiter=',', quotechar='|')
        for each in data1:
            X = []
            for x in each:
                #print x
                if (x=="vhigh" or x == "5more" or x == "more"):
                    x=3
                elif (x=="high" or x == "big" or x == "4"):
                    x=2
                elif (x == "med" or x == "3"):
                    x=1
                elif (x == "low" or x == "small" or x == "2"):
                    x=0
                X.append(x)
            if (X[-1] == "acc"):
                for i in range(3):
                    data.append(X)
            elif (X[-1] == "good"):
                for i in range(17):
                    data.append(X)
            elif (X[-1] == "vgood"):
                for i in range(18):
                    data.append(X)
            else:
                data.append(X)
    shuffle(data)
    size = int(len(data) * 0.8)
    train_data = data[0:size]
    test_data = data[size:len(data)]
    train_class = np.array(train_data)[:, len(train_data[0]) - 1]
    train_data = np.array(train_data)[:, range(0, len(train_data[0]) - 1)]
    test_class = np.array(test_data)[:, len(test_data[0]) - 1]
    test_data = np.array(test_data)[:, range(0, len(test_data[0]) - 1)]
    train_data = [[int(j) for j in i] for i in train_data]
    test_data = [[int(j) for j in i] for i in test_data]

        
def main():
    LoadData1("car.data.txt")
    Boosting()
    Bagging()
    classification()
    
    print('main')
    
if __name__ == "__main__":main()
