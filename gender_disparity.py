# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:32:11 2023

@author: new
"""
import pandas as pd
#from pandas_profiling import ProfileReport
import numpy as np
# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/xAPI-Edu-Data.csv')
columns = list(data)
"""
['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 
 'Topic', 'Semester', 'Relation', 'raisedhands', 'VisITedResources', 
 'AnnouncementsView', 'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 
 'StudentAbsenceDays', 'Class']
"""

#DIMENSIONALITY REDUCTION
data = data.drop(['NationalITy', 'PlaceofBirth'], axis=1)

# Handle categorical data
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])
data['StageID'] = encoder.fit_transform(data['StageID'])
data['GradeID'] = encoder.fit_transform(data['GradeID'])
data['SectionID'] = encoder.fit_transform(data['SectionID'])
data['Topic'] = encoder.fit_transform(data['Topic'])
data['Semester'] = encoder.fit_transform(data['Semester'])
data['Relation'] = encoder.fit_transform(data['Relation'])
data['ParentAnsweringSurvey'] = encoder.fit_transform(data['ParentAnsweringSurvey'])
data['ParentschoolSatisfaction'] = encoder.fit_transform(data['ParentschoolSatisfaction'])
data['StudentAbsenceDays'] = encoder.fit_transform(data['StudentAbsenceDays'])
data['Class'] = encoder.fit_transform(data['Class'])
    
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="YlOrRd")
#Identify our variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# FEATURE SCALING
scaler = StandardScaler()
X = X.values
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function for plotting the Confusion matrix
def confMatrix(class_labels, name):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.get_cmap('Reds'))
    plt.colorbar(cax)
    ax.set_xticklabels([''] + class_labels)
    ax.set_yticklabels([''] + class_labels)
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='k')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(name)
    plt.show()
    
# TRain our model
# BASE MODEL 1
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_train_pred = tree.predict(X_train) # predicted y_train
# Call the confusion matrix method
name = "Decision Tree"
cm = confusion_matrix(y_test, tree_pred)
accuracy = accuracy_score(y_test, tree_pred)
precision = precision_score(y_test, tree_pred, pos_label=1, average='macro')
recall = recall_score(y_test, tree_pred, pos_label=1, average='macro')
f1 = f1_score(y_test, tree_pred, average='macro')
print(name)
print("tree Accuracy :", accuracy)
print("tree Precision :", precision)
print("tree Recall :", recall)
print("tree F1 :", f1)
class_labels = ["L", "M", "H"]
confMatrix(class_labels, name)
"""Decision tree result 
Decision Tree
tree Accuracy : 0.7152777777777778
tree Precision : 0.726010550023708
tree Recall : 0.7139655428802073
tree F1 : 0.7195999402895955 """

# BASE MODEL 2
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_train_pred = svm.predict(X_train) # predicted y_train
# Call the confusion matrix method
name = "Support Vector Machine"
cm = confusion_matrix(y_test, svm_pred)
accuracy = accuracy_score(y_test, svm_pred)
precision = precision_score(y_test, svm_pred, pos_label=1, average='macro')
recall = recall_score(y_test, svm_pred, pos_label=1, average='macro')
f1 = f1_score(y_test, svm_pred, average='macro')
print(name)
print("svm Accuracy :", accuracy)
print("svm Precision :", precision)
print("svm Recall :", recall)
print("svm F1 :", f1)
class_labels = ["L", "M", "H"]
confMatrix(class_labels, name)
""" Support Vector Machine result 
Support Vector Machine
svm Accuracy : 0.7569444444444444
svm Precision : 0.7599099099099099
svm Recall : 0.7588001614513628
svm F1 : 0.7591217579775932 """


# BASE MODEL 3
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
log_pred = logReg.predict(X_test)
log_train_pred = logReg.predict(X_train)
# Call the confusion matrix method
name = "Logistic Regression"
cm = confusion_matrix(y_test, log_pred)
print(cm)
accuracy = accuracy_score(y_test, log_pred)
precision = precision_score(y_test, log_pred, pos_label=1, average='micro')
recall = recall_score(y_test, log_pred, pos_label=1, average='micro')
f1 = f1_score(y_test, log_pred, average='micro')
print(name)
print("LR Accuracy :", accuracy)
print("LR Precision :", precision)
print("LR Recall :", recall)
print("LR F1 :", f1)
class_labels = ["L", "M", "H"]
confMatrix(class_labels, name)
""" Logistic regression result 
Logistic Regression
LR Accuracy : 0.7708333333333334
LR Precision : 0.7708333333333334
LR Recall : 0.7708333333333334
LR F1 : 0.7708333333333333 """



# BASE MODEL 4
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_train_pred = knn.predict(X_train) #  predicted y_train
# Evaluate the model
name = "K NEAREST NEIGHBOR"
cm = confusion_matrix(y_test, knn_pred)
accuracy = accuracy_score(y_test, knn_pred)
precision = precision_score(y_test, knn_pred, pos_label=1, average='macro')
recall = recall_score(y_test, knn_pred, pos_label=1, average='macro')
f1 = f1_score(y_test, knn_pred, pos_label=1, average='macro')
print(name)
print("knn Accuracy :", accuracy)
print("knn Precision :", precision)
print("knn Recall :", recall)
print("knn F1 :", f1)
class_labels = ["L", "M", "H"]
confMatrix(class_labels, name)
""" KNN result
K NEAREST NEIGHBOR
knn Accuracy : 0.6944444444444444
knn Precision : 0.6876846803407392
knn Recall : 0.7292786381435926
knn F1 : 0.7007146393579321 """



""" SUPER MODEL """
forest = RandomForestClassifier()
""" Start training """
forest.fit(X_train, y_train)
forest.fit(X_train, tree_train_pred)
forest.fit(X_train, svm_train_pred)
forest.fit(X_train, log_train_pred)
forest.fit(X_train, knn_train_pred)
""" End of training """
forest_pred = forest.predict(X_test)
forest_train_pred = forest.predict(X_train) #  predicted y_train
# Call the confusion matrix method
name = "Random Forest"
cm = confusion_matrix(y_test, forest_pred)
accuracy = accuracy_score(y_test, forest_pred)
precision = precision_score(y_test, forest_pred, pos_label=1, average='macro')
recall = recall_score(y_test, forest_pred, pos_label=1, average='macro')
f1 = f1_score(y_test, forest_pred, average='macro')
print(name)
print("forest Accuracy :", accuracy)
print("forest Precision :", precision)
print("forest Recall :", recall)
print("forest F1 :", f1)
class_labels = ["L", "M", "H"]
confMatrix(class_labels, name)
""" Random forest result
Random Forest
forest Accuracy : 0.7708333333333334
forest Precision : 0.7703781512605041
forest Recall : 0.7796649176102366
forest F1 : 0.7720413505898787 """

















