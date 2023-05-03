import os
import pathlib

MODEL_DICT = {
    1:'01_RandomForest.py',       
    2:'02_LogisticRegression.py', 
    3:'03_Bow.py',                
    4:'04_FeatureEnginnering.py', 
    5:'05_LSTM.py',               
    6:'06_BERT.py'
}

if __name__ == '__main__':
    print('Welcome to use the Scam Job Post Detection Model Tester.')
    print('Note: \
        \n 1. Please make sure you have trained the model before testing. \
        \n 2. This is still a simple version of the tester. It does not accept any input. \
        \n 3. To customize the test set, please modify Data/test.csv. \
        \n 4. There are pre-trained model provided for BERT and LSTM, please check Model/BERT_Model and Model/LSTM_Model. \
        \n 5. If you want to test the pre-trained model, please enter y when asked. \
        \n 6. Model 1 and 2 are trained on a balanced set while the rest are trained on an imbalanced set. \
        \n====================================')
    model = int(input('Which model do you want to test? \
            \n 1: Random Forest, \
            \n 2: Logistic Regression,  \
            \n 3: BoW, \
            \n 4: Feature Engineering + Logistic Regression, \
            \n 5: LSTM, \
            \n 6: BERT \
            \nPlease enter the number of the model:' 
                  ))
    print('====================================')
    print('Testing model: ', MODEL_DICT[model])
    cur_dir = pathlib.Path(__file__).parent.resolve()
    os.chdir(cur_dir)
    command = 'python ' +MODEL_DICT[model] + ' -test'
    os.system(command)
    
