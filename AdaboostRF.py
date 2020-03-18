# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 04:50:50 2020

@author: Aryan
"""


from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import keras
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import keras
import keras.backend as K
from keras.layers import Input,Dropout, Flatten,Dense, Activation,MaxPooling1D
from keras.layers import dot, multiply, concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix 

path = "A:\Project\Attention\Data"
file1 = "gatedAtnAll_Input.csv"
file2 = "STACKED_RF.csv"
file3 = "TCGA_gatedAtnAll_Input.csv"
file4 = "TCGA_STACKED_RF.csv"
#dataset1 = np.loadtxt(os.path.join(path, file1),delimiter=",")
#dataset1 = np.loadtxt(os.path.join(path, file2),delimiter=",")
#dataset1 = np.loadtxt(os.path.join(path, file3),delimiter=",")
dataset1 = np.loadtxt(os.path.join(path, file4),delimiter=",")

#X1= dataset1[:,0:775]
#Y1 = dataset1[:,775]
#X1= dataset1[:,0:450]
#Y1 = dataset1[:,450]
#X1= dataset1[:,0:761]
#Y1 = dataset1[:,761]
X1= dataset1[:,0:450]
Y1 = dataset1[:,450]










#x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)
rfc = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0,class_weight='balanced')
#rf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None,min_samples_split=0.05,min_samples_leaf=0.4,max_features=0.4),n_estimators=200) 
rf = AdaBoostClassifier(base_estimator=rfc,n_estimators=200) 

scores1 = cross_val_score(rf, X1, Y1, cv=10,verbose=0)

print ("Cross-validated scores:", scores1)
print("Accuracy = %.3f%% (+/- %.3f%%)\n" % (np.mean(scores1), np.std(scores1)))

predictions1 = cross_val_predict(rf, X1, Y1, cv=10,method='predict_proba')[:, 1]


fpr, tpr, thresholds = roc_curve(Y1, predictions1,pos_label=1)

roc_auc = auc(fpr, tpr)
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    print('AUC:', roc_auc)

evaluate_threshold(0.45)
'''for threshold in np.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)'''



