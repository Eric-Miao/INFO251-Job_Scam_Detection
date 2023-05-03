import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from string import punctuation



dfRaw = pd.read_csv('./../Data/emscad_v1.csv')

dfRaw.fillna(' ', inplace=True)

def combineText(df):
    col = [ 'company_profile','description', 'requirements', 'benefits', 
         'title', 'location', 'employment_type',
        'required_experience', 'required_education', 'industry', 'function']
    text = ''
    for c in col:
        if df[c] not in text:
            text+=df[c]
            text+=' '
    return text

dfRaw['text'] = dfRaw.apply(lambda x: combineText(x),axis=1)


def replaceAbb(string):
    contractions = {
        "n't": " not",
        "'ve": " have",
        "cause": "because",
        "'d": " would",
        "'ll": " will",
        "'s": " is",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "'re": " are",
  }
    for abb, full in contractions.items():
        string = string.replace(abb, full)
    return string

def removeTag(string):
    patternTag = re.compile(r'(<.*?>)')
    return patternTag.sub('',string)

def removeChar(string):
    replacement = [u"\xa0", "http", "https", "://",".",
                   "•","‘","’", "-","–","\r","\n","|","/",
                  "…","+",'`','"','\'',"“","”",","]
    for char in replacement:
        string = string.replace(char,' ')
    string = re.sub(' +', ' ',string)
    return string
    # patternChar = re.compile(r'(\\\w+)')
    # return patternChar.sub('',string)


dfText = dfRaw[['text','fraudulent']].copy()
dfText.text = dfText.text.str.lower()
dfText.text = dfText.text.apply(lambda x: removeChar(removeTag(replaceAbb(x))))

stop_words = stopwords.words('english')
punc_list = list(punctuation)
def removePuncStopword(string):

    tokens = word_tokenize(string)
    clean_tokens = [token for token in tokens if token not in stop_words and token not in punc_list]
    clean_tokens = [token for token in clean_tokens if 'url_' not in token and 'email_' not in token]
    return ' '.join(clean_tokens)

dfText.text = dfText.text.apply(lambda x: removePuncStopword(x))

dfText.to_csv('./../Data/text.csv',index=False)


dfText.fraudulent = dfText.fraudulent.map({'t':1,'f':0})
dfTrain,dfTest = train_test_split(dfText,test_size=0.25,random_state=42)

dfTrain.to_csv('./../Data/train.csv',index=False)
dfTest.to_csv('./../Data/test.csv',index=False)

print('Train,test data splited!')




df1 = dfText[dfText['fraudulent']==0]
df1 = df1.sample(1000)
df2 = dfText[dfText['fraudulent']==1]

df_balanced = pd.concat([df1,df2],axis=0)

df_balanced = df_balanced.sample(frac=1) #shuffle all rows
df_balanced.value_counts()
dfTrain_Balanced,dfTest_Balanced = train_test_split(df_balanced,test_size=0.25,random_state=42)


dfTrain_Balanced.to_csv('./../Data/train_balanced.csv',index=False)
dfTest_Balanced.to_csv('./../Data/test_balanced.csv',index=False)

print('Balanced Train,test data splited!')






