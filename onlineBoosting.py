from random import shuffle
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
import csv
from collections import Counter
import time 

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
    f1=f1_score(y_true, y_pred, average='macro')
    return f1
def precision(y_true,y_pred):
    pre = precision_score(y_true, y_pred, average='macro')
    return pre
def recall(y_true,y_pred):
    recall = recall_score(y_true, y_pred, average='macro')
    return recall


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
                if (x == "vhigh" or x == "5more" or x == "more"):
                    x=3
                elif (x == "high" or x == "big" or x == "4"):
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

def addModels():
    global models
    global M
    for i in range(0,M):
        models.append(linear_model.Perceptron())#linear_model.Perceptron()//class_weight="balanced"

def fit(data,classdata):
    global models
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
            models[i].partial_fit(data, classdata, classes=["vgood","good","acc","unacc"])
        prediction = models[i].predict(data)
        if compare(prediction, classdata):
            correctWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                       (2 * correctWeight[i])
        else:
            wrongWeight[i] += lam
            lam *= (correctWeight[i] + wrongWeight[i]) / \
                       (2 * wrongWeight[i])
        #print("For" + str(i) + ' Accuracy %f' % (f1score(classdata, prediction)))

def initial_fit(data,classdata):
    global models
    global train_class
    global M
    global wrongWeight
    global correctWeight
    global epsilon

    lam = 1.0
    for i in range(0, M):
        models[i].partial_fit(data, classdata, classes=["vgood","good","acc","unacc"])
        
def predict(test_data):
    prediction = []
    for i in range(0, 100):
        prediction.append(models[i].predict(test_data))
    prediction = np.array(prediction).transpose()
    Final = []
    for each in prediction:
        Final.append(Counter(each).most_common(1)[0][0])
    #print (test_class, Final)
    print ("Precision is ", precision(test_class, np.array(Final)))
    print ("Recall is ", recall(test_class, np.array(Final)))
    print ("F1 score is ", f1score(test_class, np.array(Final)))
    
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

    M = 100
    models = []
    kArr = [0]*1000
    LoadData1("./car.data.txt")
    addModels()
    wrongWeight = [0 for i in range(M)]
    correctWeight = [0 for i in range(M)]
    epsilon = [0 for i in range(M)]

    start = 0
    end = len(train_data)
    offset = 1
    count = 0
    start_time = time.time()
    while(start < end):
        if (count %400 ==0):
            print(count, 'iteration')
        count += 1
        data = train_data[start:start + offset]
        classdata = train_class[start:start + offset]
        start += offset
        fit(data, classdata)
        
    predict(test_data)
    print(time.time()-start_time)
if __name__ == "__main__":main()

