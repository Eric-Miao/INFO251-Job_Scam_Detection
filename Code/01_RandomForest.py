##################################################
## This module contains the code for Random Forest. 
## The model stores the pre-tuned optimal hyperparameters.
## If you want to review the hyperparameter tuning process, please refer to the notebooks in the folder Code/Jupyter Notebooks.
##################################################
## Author: {Yuxin Miao}
## Original Author: {Shuyao Wang}
##################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import sys
import pickle
import pandas as pd
import numpy as np
def train(X_train, y_train):
    count_vector = CountVectorizer(ngram_range=(1, 1), lowercase = True , stop_words =  'english')
    X_train = count_vector.fit_transform(X_train) 
    pickle.dump(count_vector, open('./../Model/RF_count_vector.pkl', 'wb'))
    best_max_depth = 30
    best_min_samples_split = 10
    best_n_estimators = 50
    model = RandomForestClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split, n_estimators=best_n_estimators)
    model.fit(X_train, y_train)
    print('Random Forest Model Trained.')
    return model


def predict(X_test,y_test,model):
    try:
        count_vector = pickle.load(open('./../Model/RF_count_vector.pkl', 'rb'))
    except:
        print('No fitted count vector found, train the model first.')
        exit() 
    X_test = count_vector.transform(X_test) 
    y_pred = model.predict(X_test)
    
    accu = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print('Random Forest Test Stats')   
    print('Accuracy Score: ', round(accu, 3))
    print('Precision Score: ', round(precision, 3))
    print('Recall Score: ', round(recall, 3))
    print ('F1 score:', round(f1, 3))
    print('====================================')
    print(y_pred)
    
def readData(mode='train'):
    if mode=='train':
        ret = readTrain()
    else:
        ret = readTest()
    return ret

def readTrain():
    return pd.read_csv('./../Data/train_balanced.csv')

def readTest():
    return pd.read_csv('./../Data/test_balanced.csv')

def saveModel(model):
    fp = './../Model/RandomForest.pkl'
    pickle.dump(model, open(fp, 'wb'))
    print('Fitted model saved')

def loadModel():
    fp = './../Model/RandomForest.pkl'
    try:
        model = pickle.load(open(fp, 'rb'))
        print('Fitted model loaded')
    except:
        print('No fitted model found')
        model = None
    return model 

if __name__ == '__main__':
    mode = sys.argv[1]
    if 'train' in mode:
        df = readData(mode='train')
        X_train = df.text 
        y_train = df.fraudulent
        model = train(X_train,y_train)
        saveModel(model)
    elif 'test' in mode:
        model = loadModel()
        if model is None:
            print('Fit the model first.')
            exit()
        df = readData(mode='test')
        X_test = df.text 
        y_test = df.fraudulent
        predict(X_test,y_test,model)