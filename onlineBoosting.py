import random as rn
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

global train_data
global train_class
global test_data
global test_class
global models
global kArr
global wrongWeight #= [0 for i in range(self.M)]
global correctWeight #= [0 for i in range(self.M)]
global epsilon #= [0 for i in range(self.M)]
global M

def f1score(y_true,y_pred):
    f1=f1_score(y_true, y_pred, average='micro')
    return f1
def precision(y_true,y_pred):
    pre=precision_score(y_true, y_pred, average='micro')
    return pre
def recall(y_true,y_pred):
    recall=recall_score(y_true, y_pred, average='micro')
    return recall


def LoadData1(path):
    global train_data
    global train_class
    global test_data
    global test_class

    data = pandas.read_csv(path, sep=",", header=None)
    print('data', data)
    data = data.replace(['vhigh', 'high', 'med', 'low',
                         '2', '3', '4', '5more', 'more',
                         'small', 'big'],
                        [int(4), int(3), 2, 1, 2, 3, 4, 6, 6, 1, 3])
    print(data)
    minority_majority_class_ratio_C = data[6].value_counts()[2] + data[6].value_counts()[3] / data[6].value_counts()[
        0] + data[6].value_counts()[1]
    print("vvvv", data[6].value_counts()[0], minority_majority_class_ratio_C)


    print('process', data)
    data = data.values.tolist()
    rn.shuffle(data)

    # deviding the data in to test set and training set
    size = int(len(data) * 0.8)
    train_data = data[0:size]
    test_data = data[size:len(data)]

    train_class = np.array(train_data)[:, len(train_data[0]) - 1]
    print(train_class, "\n")
    train_data = np.array(train_data)[:, range(0, len(train_data[0]) - 1)]

    train_data = [[int(j) for j in i] for i in train_data]

    print(isinstance(train_data[0][0], int))
    test_class = np.array(test_data)[:, len(test_data[0]) - 1]
    test_data = np.array(test_data)[:, range(0, len(test_data[0]) - 1)]
    test_data = [[int(j) for j in i] for i in test_data]

def addModels():
    global models
    global M
    for i in range(0,M):
        models.append(linear_model.SGDClassifier())#linear_model.Perceptron()//class_weight="balanced"

def fit(data,classdata):
    global models
    global kArr
    global train_class
    global M
    global wrongWeight
    global correctWeight
    global epsilon

    lam = 1.0
    for i in range(0, M):
        k = np.random.poisson(lam)
        if not k:
            continue
        for j in range(0,k):
            models[i].partial_fit(data,classdata)
        prediction = models[i].predict(data)
        if compare(prediction, classdata):
            correctWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                       (2 * correctWeight[i])
        else:
            wrongWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                       (2 * wrongWeight[i])
        print("For" + str(i) + ' Accuracy %f' % (f1score(classdata, prediction)))

def initial_fit(data,classdata):
    global models
    global kArr
    global train_class
    global M
    global wrongWeight
    global correctWeight
    global epsilon

    lam = 1.0
    for i in range(0, M):
        k = np.random.poisson(lam)
        if not k:
            k+=1
        for j in range(0, k):
            models[i].fit(data, classdata)
        prediction = models[i].predict(data)
        if compare(prediction,classdata):
            correctWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                   (2 * correctWeight[i])
        else:
            wrongWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                   (2 * wrongWeight[i])
        print("For" + str(i) + ' Accuracy %f' % (f1score(classdata, prediction)))
def predict(test_data):
    global M
    for i in range(0, M):
        prediction = models[i].predict(test_data)
        print('prediction and leng and test_cls len',len(prediction),len(test_class))
        print("For"+ str(i) + ' Accuracy %f' % (f1score(test_class, prediction)))
def compare(data,classdata):
    if(len(data)!=len(classdata)):
        return False
    else:
        for i in range(0,len(data)):
            if(data[i]!=classdata[i]):
                return False
    return True


# def predict(self, features):
#     label_weights = defaultdict(int)
#     for i in range(0, M):
#         epsilon = (correctWeight[i] + 1e-16) / \
#                       (wrongWeight[i] + 1e-16)
#         weight = log(epsilon)
#         label = models[i].predict(features)
#         label_weights[label] += weight
#     return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))

def main():
    global models
    global  kArr
    global M
    global wrongWeight
    global correctWeight
    global epsilon

    M = 10
    models = []
    kArr = [0]*100
    LoadData1("./car.data.txt")
    addModels()
    wrongWeight = [0 for i in range(M)]
    correctWeight = [0 for i in range(M)]
    epsilon = [0 for i in range(M)]

    start = 0
    end = len(train_data)
    offset = 200
    data = train_data[start:start + offset]
    classdata = train_class[start:start + offset]
    offset=20
    start += offset
    print(1,'iteration')
    print('\n')
    count = 2
    initial_fit(data, classdata)
    while(start <= end):
        print(count, 'iteration')
        print('\n')
        count+=1
        data = train_data[start:start + offset]
        classdata = train_class[start:start + offset]
        start += offset
        fit(data,classdata)
    predict(test_data)
    print('kArr',kArr)

if __name__ == "__main__":main()

