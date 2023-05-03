##################################################
## This module contains the code for BERT model. 
## The model stores the pre-tuned optimal hyperparameters.
## If you want to review the hyperparameter tuning process, please refer to the notebooks in the folder Code/Jupyter Notebooks.
##################################################
## Author: {Yuxin Miao}
## Original Author: {Yuxin Miao}
##################################################

import pandas as pd
import numpy as np
import keras
import os
import re
import math
import shutil
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer


from keras.metrics import AUC,Recall,Precision,BinaryAccuracy
import keras.backend as K
from keras.layers import Hashing
from keras.utils import pad_sequences
tf.get_logger().setLevel('ERROR')

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

class MyBERT():
  def __init__(self,bert_model_name='small_bert/bert_en_uncased_L-4_H-512_A-8'):
    self.bert_model_name = bert_model_name
    
    self.tfhub_handle_encoder = map_name_to_handle[self.bert_model_name]
    self.tfhub_handle_preprocess = map_model_to_preprocess[self.bert_model_name]

    print(f'BERT model selected           : {self.tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {self.tfhub_handle_preprocess}')
    self.bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
    self.bert_model = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')

  def build_classifier_model(self,r_dropout=0.1):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = self.bert_preprocess_model
    encoder_inputs = preprocessing_layer(text_input)
    encoder = self.bert_model
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(r_dropout)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

  def fit(self,X_train,y_train,class_wight=[1.0,1.0]):
    def F1Score(y_true, y_pred): #taken from old keras source code
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
      precision = true_positives / (predicted_positives + K.epsilon())
      recall = true_positives / (possible_positives + K.epsilon())
      f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
      return f1_val

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = [AUC(),Recall(),Precision(),BinaryAccuracy(),F1Score]
    epochs = 15
    steps_per_epoch = len(X_train)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    self.classifier_model = self.build_classifier_model()
    self.classifier_model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metric)

    print(f'Training model with {self.tfhub_handle_encoder}')
    history = self.classifier_model.fit(x=X_train,y=y_train,
                                  validation_split=0.2,
                                  class_weight={0:class_wight[0],1:class_wight[1]},
                                  epochs=epochs)
    return history

  def eval(self,X_test,y_test):
    return self.classifier_model.evaluate(x=X_test,y=y_test)

  def save(self):
    saved_model_path = './../Model/BERT_MODEL/newBert'
    self.classifier_model.save(saved_model_path, include_optimizer=False)
    print(f'model {saved_model_path} saved!')


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


def train(X_train,y_train,m=None,r=None,w=None):
    if m == None:
        m = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    else:
        if m not in map_name_to_handle.keys():
           print('Model Name Error: Please input a valid pre-trained BERT model name.')
           return -1
    
    if r == None:
       r = 0.2
    else:
        if not isinstance(r,float):
            print('Droprate Error: Please input a droprate between 0 and 1.')
            return -1
    
    if w == None:
       w = [1.0,2.0]
    else:
        if not isinstance(w,list):
            print('Class Weight Error: Please input a list of 2 float number as the weight of class 0 and 1 in training.')
            return -1
        

    model = MyBERT(bert_model_name=m)
    model.build_classifier_model(r)
    model.fit(X_train,y_train,w)
    model.save()

def loadModel(pre_trained=True):
    if pre_trained:
        fp = './../Model/BERT_MODEL/pre-trained-BERT'
    else:
        fp = './../Model/BERT_MODEL/newBert'
    def F1Score(y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    
    # reloaded_model = tf.keras.saving.load_model(fp,custom_objects={'F1Score':F1Score})
    reloaded_model =  tf.keras.models.load_model(fp,custom_objects={'F1Score':F1Score})
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = [AUC(),Recall(),Precision(),BinaryAccuracy(),F1Score]
    epochs = 15
    steps_per_epoch = 13410
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                        num_train_steps=num_train_steps,
                        num_warmup_steps=num_warmup_steps,
                        optimizer_type='adamw')

    reloaded_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metric)
    
    return reloaded_model

def predict(X_test,y_test,model):
    y_pred = model.predict(X_test)
    def sig(x):
        return 1/(1 + np.exp(-x))
    y_pred = np.round(sig(y_pred)).astype(int)
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
        df = readData(mode='train')
        X_train = df.text 
        y_train = df.fraudulent
        train(X_train,y_train,m=None,r=None,w=None)
    elif 'test' in mode:
        df = readData(mode='test')
        X_test = df.text 
        y_test = df.fraudulent

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