
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import sys
import pickle
import pandas as pd
import numpy as np


def method_one_text_features(df2, train_or_test):
    df_new = df2.copy()
    df_new['company_profile'] = df_new['company_profile'].str.lower()
    df_new['company_profile_count_chr'] = df_new['company_profile'].str.len()
    df_new['company_profile_count_word'] = df_new['company_profile'].str.split(' ').str.len()
    
    df_new['description'] = df_new['description'].str.lower()
    df_new['description_profile_count_chr'] = df_new['description'].str.len()
    df_new['description_profile_count_word'] = df_new['company_profile'].str.split(' ').str.len()
    
    df_new['requirements'] = df_new['requirements'].str.lower()
    df_new['requirements_count_chr'] = df_new['requirements'].str.len()
    df_new['requirements_profile_count_word'] = df_new['company_profile'].str.split(' ').str.len()
    
    df_new['benefits'] = df_new['benefits'].str.lower()
    df_new['benefits_count_chr'] = df_new['benefits'].str.len()
    df_new['benefits_profile_count_word'] = df_new['company_profile'].str.split(' ').str.len()

    df_new['excl_count'] = df_new['company_profile'].str.count('!') + df_new['description'].str.count('!') + df_new['requirements'].str.count('!') + df_new['benefits'].str.count('!')
    df_new['q_count'] =  df_new['company_profile'].str.count('\?') + df_new['description'].str.count('\?') + df_new['requirements'].str.count('\?') + df_new['benefits'].str.count('\?') 
    df_new['hash_count'] = df_new['company_profile'].str.count('#') + df_new['description'].str.count('#') + df_new['requirements'].str.count('#') + df_new['benefits'].str.count('#')
    df_new['dollar_count'] = df_new['company_profile'].str.count('$') + df_new['description'].str.count('$') + df_new['requirements'].str.count('$') + df_new['benefits'].str.count('$')
    df_new['newline_count'] = df_new['company_profile'].str.count('\n') + df_new['description'].str.count('\n') + df_new['requirements'].str.count('\n') + df_new['benefits'].str.count('\n')
    df_new['bracket_count'] =  df_new['company_profile'].str.count(r'\<.*\>') + df_new['description'].str.count(r'\<.*\>') + df_new['requirements'].str.count(r'\<.*\>') + df_new['benefits'].str.count(r'\<.*\>')

    df_new = df_new.fillna(0)

    # dummy_df = pd.get_dummies(data=df_new, columns=['telecommuting','has_company_logo', 'has_questions','employment_type','required_experience','required_education','industry','function'])
    # df_new = pd.concat([df_new.drop(columns=['telecommuting','has_company_logo', 'has_questions','employment_type','required_experience', 'required_education','industry','function']), dummy_df], axis=1)
    # print(list(df_new.columns))
    
    if train_or_test == 'train':
        df_new = df_new.drop(['telecommuting', 'has_company_logo', 'has_questions', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function','title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits', 'in_balanced_dataset',
                             'company_profile_count_chr', 'description_profile_count_chr', 'requirements_count_chr', 'benefits_count_chr'], axis=1)
    else: 
        df_new = df_new.drop(['telecommuting', 'has_company_logo', 'has_questions', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function','title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits', 'in_balanced_dataset',
                             'company_profile_count_chr', 'description_profile_count_chr', 'requirements_count_chr', 'benefits_count_chr'], axis=1)
    return df_new

def train(X_train,y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model 
def predict(X_test,y_test,model):
    y_pred = model.predict(X_test)

    accu = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print('Logistic Regression with Feature Engineering Test Stats')    
    print('Accuracy Score: ', round(accu, 3))
    print('Precision Score: ', round(precision, 3))
    print('Recall Score: ', round(recall, 3))
    print ('F1 score:', round(f1, 3))
    print('====================================')
    print(y_pred)

    
def readData(mode='train'):
    master_df = pd.read_csv('./../Data/emscad_v1.csv')
    master_df_no_y = master_df.drop('fraudulent' ,axis =1)

    master_df['fraudulent']=master_df['fraudulent'].replace('f', 0)
    master_df['fraudulent']=master_df['fraudulent'].replace('t', 1)
    X_train, X_test, y_train, y_test = train_test_split(master_df_no_y, master_df['fraudulent'], test_size=0.25, random_state=42)
    if mode=='train':
        X = method_one_text_features(X_train, 'train')
        y = np.array(y_train)
    else:
        X = method_one_text_features(X_test, 'test')
        y = np.array(y_test)
    return (X,y)

# def readTrain():
#     return pd.read_csv('./../Data/train_balanced.csv')

# def readTest():
#     return pd.read_csv('./../Data/test_balanced.csv')


def saveModel(model):
    fp = './../Model/FeatureEnginnering_RF.pkl'
    pickle.dump(model, open(fp, 'wb'))
    print('Fitted model saved')

def loadModel():
    fp = './../Model/FeatureEnginnering_RF.pkl'
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
        X_train,y_train = readData(mode='train')
        model = train(X_train,y_train)
        saveModel(model)
    elif 'test' in mode:
        model = loadModel()
        if model is None:
            print('Fit the model first.')
            exit()
        X_test,y_test = readData(mode='test')
        predict(X_test,y_test,model)