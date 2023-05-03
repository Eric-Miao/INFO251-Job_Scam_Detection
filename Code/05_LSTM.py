import pandas as pd
import numpy as np
import keras
import os
import re
import math
# from google.colab import drive
import sys

from keras.layers import Hashing
from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

from keras.layers import Embedding,LSTM,Bidirectional,Dropout,Dense
from keras.models import Sequential
import random
import numpy as np
import tensorflow as tf
seed=42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

BEST_EMBEDING = 50
BEST_LSTM_LAYER = 3
BEST_WEIGHT = [1.0,2.0]
MAX_SEQUENCE_LENGTH = 500
n_words = 43428

# textRaw = dfRaw.text.apply(lambda x: textProcessing(x)).to_list()
def textProcessing(string):
  ret = re.sub('[^a-zA-Z]',' ',string).split()
  ret = ' '.join(ret)
  return ret

def oneHot(text):
    unique = set()
    for t in text:
        unique.update(set(t.split()))
    unique = list(unique)
    onehot_hasher = Hashing(num_bins=n_words,output_mode='one_hot')
    onehot_map = onehot_hasher(unique)
    onehot_dict = dict(zip(unique,onehot_map))
    # turn pure onthot into onehot index for keras embeding layer
    textOnehot = [np.array([int(onehot_dict[x].numpy().argmax()) for x in t.split()]) for t in text]
    X = pad_sequences(textOnehot, maxlen=MAX_SEQUENCE_LENGTH)
    return X

def readData(mode='train'):
    if mode=='train':
        ret = readTrain()
        textRaw = ret.text.apply(lambda x: textProcessing(x)).to_list()
        X = oneHot(textRaw)
        y = ret.fraudulent.to_numpy()
    else:
        ret = readTest()
        textRaw = ret.text.apply(lambda x: textProcessing(x)).to_list()
        X = oneHot(textRaw)
        y = ret.fraudulent.to_numpy()
    return (X,y)

def readTrain():
    return pd.read_csv('./../Data/train.csv')

def readTest():
    return pd.read_csv('./../Data/test.csv')


def train(X_train,y_train,embeding_size=None,LSTM_layer=None,w=None):
    if embeding_size == None:
        embeding_size = BEST_EMBEDING

    if LSTM_layer == None:
       LSTM_layer = BEST_LSTM_LAYER
    else:
        if not isinstance(LSTM_layer,int):
            print('LSTM Layer Error: Please input a extra LSTM layer number starting from 0.')
            return -1
    
    if w == None:
       w = BEST_WEIGHT
    else:
        if not isinstance(w,list):
            print('Class Weight Error: Please input a list of 2 float number as the weight of class 0 and 1 in training.')
            return -1
        

    embedding_vector_features_size=embeding_size
    LSTMmodel=Sequential()
    LSTMmodel.add(Embedding(n_words,embedding_vector_features_size,input_length=MAX_SEQUENCE_LENGTH))
    for _ in range(LSTM_layer-1):
        LSTMmodel.add(Bidirectional(LSTM(100, return_sequences=True)))
    LSTMmodel.add(Bidirectional(LSTM(100)))
    LSTMmodel.add(Dropout(0.1))
    LSTMmodel.add(Dense(1,activation='sigmoid'))
    LSTMmodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['AUC','Recall','Precision'])
    print(f'model initialized')
    model = LSTMmodel
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        class_weight = {0:w[0],1:w[1]},
        verbose=1
    )
    print(f'model fitted')
    saved_model_path = './../Model/LSTM_MODEL/newLSTM'
    model.save(saved_model_path, include_optimizer=False)

def loadModel(pre_trained=True):
    if pre_trained:
        fp = './../Model/LSTM_MODEL/pre-trained-LSTM'
    else:
        fp = './../Model/LSTM_MODEL/newLSTM'

    
    # reloaded_model = tf.keras.saving.load_model(fp)
    reloaded_model =  tf.keras.models.load_model(fp)


    reloaded_model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['AUC','Recall','Precision']
                           )
    
    return reloaded_model

def predict(X_test,y_test,model):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    accu = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print('BERT Test Stats')   
    print('Accuracy Score: ', round(accu, 3))
    print('Precision Score: ', round(precision, 3))
    print('Recall Score: ', round(recall, 3))
    print ('F1 score:', round(f1, 3))
    print('====================================')
    y_pred = np.reshape(y_pred,(-1,1))
    print(*y_pred)


if __name__ == '__main__':
    mode = sys.argv[1]
    if 'train' in mode:
        X_train,y_train = readData(mode='train')
        train(X_train,y_train,embeding_size=None,LSTM_layer=None,w=None)
    elif 'test' in mode:
        X_test,y_test = readData(mode='test')

        pre_trained = input('Input y if you want to test with the pretrained BERT model.\nInput anything else for your newly trained model.\nInput: ')
        if pre_trained == 'y':
            model = loadModel(pre_trained=True)
        else:
            if 'newBert' not in os.listdir('./../Model/BERT_MODEL'):
                print('No newly trained model, test with pre-trained model.')
                model = loadModel(pre_trained=True)
            else:
                model = loadModel(pre_trained=False)
        predict(X_test,y_test,model)