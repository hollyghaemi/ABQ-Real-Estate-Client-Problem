#%% import packages

# Numpy, pandas, random, matplotlib & sklearn  packages added.
import numpy as np
import pandas as pd

import random as rn

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

import itertools


import os

#%% Set current directory and perform Exploratory Data Analysis on 'hdma_abq.'
os.chdir("/Users/hallehghaemi/Desktop/RS21")
df = pd.read_excel('hdma_abq.xlsx')
df.head()
df.shape
df.info()
list(df)
df.describe()
df.county_name.unique() #variable we are trying to predict
df.county_name.value_counts()
county_names = list(df.county_name.unique())
county_names_vector = df['county_name'] #saving before encoding for EDA in next script
df.corr()

#%% Clean out NAs in dataframe with mode imputation
print(df.isnull().sum()) #column-wise distribution null values
df = df.fillna(df['tract_to_msamd_income'].value_counts().index[0])
df = df.fillna(df['rate_spread'].value_counts().index[0])
df = df.fillna(df['population'].value_counts().index[0])
df = df.fillna(df['minority_population'].value_counts().index[0])
df = df.fillna(df['number_of_owner_occupied_units'].value_counts().index[0])
df = df.fillna(df['number_of_1_to_4_family_units'].value_counts().index[0])
df = df.fillna(df['applicant_income_000s'].value_counts().index[0])
df = df.fillna(df['census_tract_number'].value_counts().index[0])
df = df.fillna(df['denial_reason_name_3'].value_counts().index[0])
df = df.fillna(df['denial_reason_name_2'].value_counts().index[0])
df = df.fillna(df['denial_reason_name_1'].value_counts().index[0])
df = df.fillna(df['applicant_race_name_5'].value_counts().index[0])
df = df.fillna(df['applicant_race_name_3'].value_counts().index[0])
df = df.fillna(df['applicant_race_name_2'].value_counts().index[0])
print(df.isnull().values.sum()) #total number of null

#%% Encoding categorical data
# Decided to use scikit-learn's LabelEncoder over one-hot encoding as to avoid 
#having too many variables. 
lb_make = LabelEncoder()

#dataframe with county name specific info, will be a translator
df_county_names = df.copy()
df_county_names['county_name_code'] = lb_make.fit_transform(df_county_names['county_name'])
df_county_names = df_county_names[['sequence_number', 'county_name', 'county_name_code']]
df_county_names.head()

# Make a separate data frame for categorical data to make an var_mod list
cat_df = df.select_dtypes(include=['object']).copy()
cat_df.head()
var_mod = list(cat_df)
print(var_mod)
df[var_mod]=df[var_mod].astype(str)

#encoding categorical variables loop
for i in var_mod:
    df[i] = lb_make.fit_transform(df[i])
df.dtypes
df.info()
df.head()

#dataframe has been filled out completely and the categorical variables
#have been encoded


