#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:36:57 2019

@author: hallehghaemi
"""

#%% Imports
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#%% Set features
X = df[subset_feature_cols] # Features
y = df.county_name # Target variable

#%% Split data into training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#%% Standardize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%% Train data
mlp = MLPClassifier(hidden_layer_sizes=(5,5,5),max_iter=500)
mlp.fit(X_train,y_train)

#%% Accuracy, precision after predictions
predictions = mlp.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, predictions)
cnf_matrix
print(confusion_matrix(y_test,predictions))
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
NN_accuracy = metrics.accuracy_score(y_test, predictions)
print(classification_report(y_test,predictions))

#%%Confusion Matrix
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
plt.show()

print("Logistic Regression Accuracy:", logistic_accuracy, "Random Forest Accuracy:"
      , RF_accuracy, "Neural Net Accuracy:", NN_accuracy)


# =============================================================================
# Logistic regression has best accuracy with random forest being close behind. 
# Neural net had the lowest accuracy. Neural net also did better with 5 hidden
# layers as opposed to 13
# =============================================================================
