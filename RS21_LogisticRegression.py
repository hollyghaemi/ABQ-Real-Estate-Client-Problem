#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:56:11 2019

@author: hallehghaemi
"""
#%% Import models from scikit learn module
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

#%% Set features
cont_mod = list(cont_df)
feature_cols = var_mod + cont_mod
feature_cols.remove('county_name')
X = df[feature_cols] # Features
y = df.county_name # Target variable

#%% Split data into training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#%% Fit model with logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#%% Run the model on the test set
y_pred=logreg.predict(X_test)

#%% Create confusion matrix & plot it with a heatmap
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1,2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#%% Review accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
logistic_accuracy = metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

#%% Conclusion, using all of the variables for the model results in overfitting
#as we have high accuracy, precision, confusion and recall values.
#I will now subset the data to only include features that the client themself
#will be able to provide

#%% Set features

remove = ['population', 'rate_spread', 'minority_population', 'number_of_owner_occupied_units',
                    'number_of_1_to_4_family_units', 'census_tract_number']
subset_feature_cols = list(set(feature_cols).difference(set(remove)))

X = df[subset_feature_cols] # Features
y = df.county_name # Target variable

#%% Split data into training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#%% Fit model with logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#%% Run the model on the test set
y_pred=logreg.predict(X_test)

#%% Create confusion matrix & plot it with a heatmap
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1,2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#%% Review accuracy, precion, recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#accuracy stayed almost the same as did precision