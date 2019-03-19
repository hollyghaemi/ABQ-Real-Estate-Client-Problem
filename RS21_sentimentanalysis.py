#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:09:54 2019

@author: hallehghaemi
"""

#%% Read in data

os.chdir("/Users/hallehghaemi/Desktop/RS21")
twitter_master = pd.read_excel('Twitter_141103.xlsx')
twitter_master.shape
twitter_master.head()


        
train = content[1:765]
test = content[766:2184]

import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_twitter(twitter):
    twitter = [REPLACE_NO_SPACE.sub("", line.lower()) for line in twitter]
    twitter = [REPLACE_WITH_SPACE.sub(" ", line) for line in twitter]
    
    return twitter

train_clean = preprocess_twitter(train)
test_clean = preprocess_twitter(test)
list(train_clean)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_clean)
X = cv.transform(train_clean)
X_test = cv.transform(test_clean)
