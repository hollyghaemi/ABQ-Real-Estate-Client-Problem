#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:20:25 2019

@author: hallehghaemi
"""
#%% Import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#%% Set features
X = df[subset_feature_cols] # Features
y = df.county_name # Target variable

#%% Split data into training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#%% Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#%% Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

#%%Test the model
y_pred=clf.predict(X_test)

#%% Model measurements
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
RF_accuracy = metrics.accuracy_score(y_test, y_pred)
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

#%% Look for important features
feature_imp = pd.Series(clf.feature_importances_,index=subset_feature_cols).sort_values(ascending=False)
feature_imp

#create a bargraph to easily visualize
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# =============================================================================
# Conclusion is that the random forest classifier is not much better than the 
# logistic regression model but now we see which features are important 
# for predicting the county that the customer will want to live in 
# =============================================================================
