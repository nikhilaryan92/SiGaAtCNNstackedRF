# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:01:21 2020

@author: Aryan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:25:29 2020

@author: Aryan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:51:17 2020

@author: Aryan
"""

import keras
import keras.backend as K
from keras.layers import Input,Dropout, Flatten,Dense,MaxPooling1D,multiply, concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix 

num_of_filters = 25
epochs = 70
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []
path = 'A:/Project/Attention/gatedAtnExpOutput.csv'
def shows_result(path,arr):
	with open(path, 'w') as f: # Change the path to your local system
		for item_exp in arr:
			for elem in item_exp:
				f.write(str(elem)+',')
			f.write('\n')



def sensitivity_at_specificity(specificity, **kwargs):
    def Sn(labels, predictions):
        # any tensorflow metric
        value, update_op = K.tf.metrics.sensitivity_at_specificity(labels, predictions, specificity, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in K.tf.local_variables() if 'sensitivity_at_specificity' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            K.tf.add_to_collection(K.tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value
    return Sn



# fix random seed for reproducibility
numpy.random.seed(1)

# load METABRIC EXPR dataset
#dataset_exp = numpy.loadtxt("A:/Project/Stacked Ensemble/Data/METABRIC_gene_exp_1980.txt", delimiter="\t")# Change the path to your local system
# split into input (X) and output (Y) variables
#X_exp = dataset_exp[:,0:400]
#Y_exp = dataset_exp[:,400]

X_exp = numpy.loadtxt("A:/Project/Attention/Data/TCGA/tcga_exp_final.csv", delimiter=",")# Change the path to your local system
Y_exp = numpy.loadtxt("A:/Project/Attention/Data/TCGA/5yearCutOff.txt")


# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores = []
i=1

for train_index, test_index in kfold.split(X_exp, Y_exp):
    print("*************************************************************",i,"th Fold **********************************************************************")
    i=i+1
	#Spliting the clinical data set into training and testing
    x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
    y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 	
    x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
    x_test_exp = numpy.expand_dims(x_test_exp, axis=2)

    # first exp CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=1)
    main_input_exp = Input(shape=(400,1),name='Input')
    
    conv_exp1 = Conv1D(filters=num_of_filters,kernel_size=2,strides=1,padding='same',name='Conv1D_exp1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    gatedAtnConv_exp1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    mult_1_1 = multiply([gatedAtnConv_exp1,conv_exp1])
    mult_1_1_1 = multiply([gatedAtnConv_exp1_1,conv_exp1])
    pooled_exp1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_exp1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    
    conv_exp2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='Conv1D_exp2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    gatedAtnConv_exp2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    mult_2_2 = multiply([gatedAtnConv_exp2,conv_exp2])
    mult_2_2_2 = multiply([gatedAtnConv_exp2_2,conv_exp2])
    pooled_exp2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_exp2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    conv_exp3 = Conv1D(filters=num_of_filters,kernel_size=4,strides=1,padding='same',name='Conv1D_exp3',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp3 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D3',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp3)
    gatedAtnConv_exp3_3 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D3_3',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp3)
    mult_3_3 = multiply([gatedAtnConv_exp3,conv_exp3])
    mult_3_3_3 = multiply([gatedAtnConv_exp3_3,conv_exp3])
    pooled_exp3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_3_3)
    pooled_exp3_3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_3_3_3)
    
    conv_exp4 = Conv1D(filters=num_of_filters,kernel_size=5,strides=1,padding='same',name='Conv1D_exp4',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp4 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D4',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp4)
    gatedAtnConv_exp4_4 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D4_4',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp4)
    mult_4_4 = multiply([gatedAtnConv_exp4,conv_exp4])
    mult_4_4_4 = multiply([gatedAtnConv_exp4_4,conv_exp4])
    pooled_exp4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_4_4)
    pooled_exp4_4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_4_4_4)
    
    conv_exp5 = Conv1D(filters=num_of_filters,kernel_size=6,strides=1,padding='same',name='Conv1D_exp5',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp5 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D5',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp5)
    gatedAtnConv_exp5_5 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D5_5',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp5)
    mult_5_5 = multiply([gatedAtnConv_exp5,conv_exp5])
    mult_5_5_5 = multiply([gatedAtnConv_exp5_5,conv_exp5])
    pooled_exp5 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_5_5)
    pooled_exp5_5 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_5_5_5)
    
    
    
    ''' added = Maximum()([conv_exp1,conv_exp2,conv_exp3,conv_exp4,conv_exp5])
    gatedAtnConv_exp = Conv1D(filters=num_of_filters,kernel_size=4,strides=1,padding='same',name='GatedConv1D',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(added)
    gatedAtnConv_activ_exp = Activation('relu')(gatedAtnConv_exp)
    print(gatedAtnConv_activ_exp.shape)
    # calulate the element wise matrix multiplication of conv_activ_exp and gatedAtnConv_activ_exp
    #gatedAtn = Lambda(multiply)([conv_activ_exp,gatedAtnConv_activ_exp])
    gatedAtn_exp = multiply([added,gatedAtnConv_activ_exp])
    print(gatedAtn_exp.shape)
    pooled_exp = MaxPooling1D(pool_size=2, strides=1, padding='same')(gatedAtn_exp)
    print(pooled_exp.shape)
    flat_exp = Flatten(name='Flatten')(pooled_exp)
    
    print(conv_exp1.shape)
    print(conv_exp2.shape)
    print(conv_exp3.shape)
    print(conv_exp4.shape)
    print(conv_exp5.shape)
    
    print(pooled_exp5.shape)
    print(pooled_exp1.shape)
    print(pooled_exp1_1.shape)
    print(pooled_exp2.shape)
    print(pooled_exp2_2.shape)
    print(pooled_exp3.shape)
    print(pooled_exp3_3.shape)
    print(pooled_exp4.shape)
    print(pooled_exp4_4.shape)
    print(pooled_exp5_5.shape)'''
    
    #merged = concatenate([pooled_exp1, pooled_exp2, pooled_exp3,pooled_exp4,pooled_exp5,pooled_exp1_1, pooled_exp2_2, pooled_exp3_3,pooled_exp4_4,pooled_exp5_5],name='merge',axis=1)
   
    merged = concatenate([pooled_exp1,pooled_exp1_1,pooled_exp2,pooled_exp2_2],name='merge',axis=1)
    flat_exp = Flatten(name='Flatten')(merged)
    dense_exp = Dense(150,name='dense_exp',activation='tanh',activity_regularizer=l2(0.01))(flat_exp)
    drop_final1 = Dropout(rate = 0.5)(dense_exp)
    dense_final2 = Dense(100,name='dense_final2',activation='tanh',activity_regularizer=l2(0.01))(drop_final1)
    dense_final3 = Dense(50,name='dense_final3',activation='tanh',activity_regularizer=l2(0.01))(dense_final2)
    output = Dense(1,activation='sigmoid')(dense_final3)
    model = Model(inputs=main_input_exp, outputs=output)
    plot_model(model, to_file='A:/Project/Attention/exp_gated_attention.png') # Change the path to your local system

    #model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)
    model.fit(x_train1, y_train1, epochs=epochs, batch_size=8,validation_data=(x_val1,y_val1))	
	


    scores = model.evaluate(x_test_exp, y_test_exp,verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    #print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    #print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))
    #acc_cvscores.append(scores[1] * 100)
    #Pr_cvscores.append(scores[2] * 100)
    Sn_cvscores.append(scores[1] * 100)
    #Mcc_cvscores.append(scores[4] * 100)
#print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_cvscores), numpy.std(acc_cvscores)))
#print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_cvscores), numpy.std(Pr_cvscores)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))
#print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_cvscores), numpy.std(Mcc_cvscores)))
X_exp=numpy.expand_dims(X_exp, axis=2)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)

intermediate_output = intermediate_layer_model.predict(X_exp)
shows_result(path,intermediate_output)

#X_exp=numpy.expand_dims(X_exp, axis=2)
#y_pred = model.predict(X_exp)
y_pred_prob = model.predict(X_exp)
'''y_pred_class = binarize(y_pred_prob, 0.5)
confusion = confusion_matrix(Y_exp, y_pred_class) 
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]'''
#print((TP + TN) / float(TP + TN + FP + FN))
# define a function that accepts a threshold and prints sensitivity and specificity
fpr, tpr, thresholds = roc_curve(Y_exp, y_pred_prob,pos_label=1)

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'Exp Gated-Attention = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

