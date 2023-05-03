import numpy as np
import scipy as sp
import pandas as pd

import pickle
import sys 

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# words_in_text function that would calculatethe frequnecy of the word
def words_in_texts(words, texts):
    indicator_array = 1 * np.array([texts.str.contains(word) for word in words]).T
    return indicator_array

def generate_features(df):
    eda_ham = df.loc[df['fraudulent']==0]
    eda_spam = df.loc[df['fraudulent']==1]

    num_ham, num_spam = {}, {}

    ham_split = eda_ham['text'].str.replace(r'/<[^>]*>/g', ' ').str.split()
    spam_split = eda_spam['text'].str.replace(r'/<[^>]*>/g', ' ').str.split()

    #put word frequencies in dictionaries
    for i in ham_split:
        for j in i:
            if num_ham.get(j) is None:
                num_ham[j] = 1
            num_ham[j] = num_ham[j] + 1
    for i in spam_split:
        for j in i:
            if num_spam.get(j) is None:
                num_spam[j] = 1
            num_spam[j] = num_spam[j] + 1
            
    #sorted_ham = sorted(num_ham, key = num_ham.get, reverse = True)
    sorted_spam = sorted(num_spam, key = num_spam.get, reverse = True)
    len(sorted_spam) #95402 pairs in the dictionary

    # appended the list of words by the number occurences in the spam set 
    feature = []
    for i in np.arange(1200):
        feature.append(sorted_spam[i])
    return feature
        
def train(X_train,y_train,feature):
    print('Model training...\n(This may take several minutes...)')
    num_words = [100, 200, 300, 400, 500, 600, 700, 800, 900,1000,1100,1200]
    fp = './../Model/Bow_Model/'
    for num in num_words:
        arr_feature = feature[:num]
        
        train_X = words_in_texts(arr_feature, X_train)
        train_Y = np.array(y_train)

        model_eda = LogisticRegression(max_iter=10000)
        model_eda.fit(train_X, train_Y)
        
        pickle.dump(model_eda, open(fp + str(num) + '.pkl', 'wb'))
    print('Model fitted.')
    print('Model saved.')

def loadModel():
        
    fp = './../Model/Bow_Model/' 
    num_words = [100, 200, 300, 400, 500, 600, 700, 800, 900,1000,1100,1200]
    MODELS = [pickle.load(open(fp + str(num) + '.pkl', 'rb')) for num in num_words]
    return MODELS

def predict(X_test,y_test,MODELS,feature):
    
    test_y = np.array(y_test)

    accuracy_score_lst = []
    precision_score_lst = []
    recall_score_lst = []
    f1_score_lst = []

    num_words = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    # num_words = [100, 200]
    for i,num in enumerate(num_words):
        arr_feature = feature[:num]     
        model_eda = MODELS[i]
        
        test_x = words_in_texts(arr_feature, X_test)
        pred = model_eda.predict(test_x)

        accuracy_score_lst.append(accuracy_score(pred, y_test))
        precision_score_lst.append(precision_score(pred, y_test))
        recall_score_lst.append(recall_score(pred, y_test))
        f1_score_lst.append(f1_score(pred, y_test))
        
    print('BoW Test Stats') 
    top_words_result = pd.DataFrame({'Accuracy':accuracy_score_lst,'Precision':precision_score_lst,
                            'Recall':recall_score_lst,'F1':f1_score_lst,}, index=num_words)
    print(top_words_result.round(3))
    print('====================================')
    print('Below is the prediction from the best n-features BoW model.')
    BEST_MODEL = MODELS[accuracy_score_lst.index(max(accuracy_score_lst))]
    
    print(len(BEST_MODEL.predict(test_x)))
    print(BEST_MODEL)   
    
def readData(mode='train'):
    if mode=='train':
        ret = readTrain()
    else:
        ret = readTest()
    return ret

def readTrain():
    return pd.read_csv('./../Data/train.csv')

def readTest():
    return pd.read_csv('./../Data/test.csv')


if __name__ == '__main__':
    mode = sys.argv[1]
    if 'train' in mode:
        df = readData(mode='train')
        X_train = df.text 
        y_train = df.fraudulent
        train(X_train,y_train,generate_features(df))
    elif 'test' in mode:
        model = loadModel()
        if model is None:
            print('Fit the model first.')
            exit()
        df = readData(mode='test')
        X_test = df.text 
        y_test = df.fraudulent
        
        df2 = readData(mode='train')
        predict(X_test,y_test,model,generate_features(df2))