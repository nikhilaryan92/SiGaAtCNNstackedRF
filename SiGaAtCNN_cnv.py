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
from keras.layers import Input,Dropout,Flatten,Dense,MaxPooling1D
from keras.layers import multiply, concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 


num_of_filters = 25
epochs = 25
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []
path = 'A:/Project/Attention/gatedAtnCnvOutput.csv'
def shows_result(path,arr):
	with open(path, 'w') as f: # Change the path to your local system
		for item_clinical in arr:
			for elem in item_clinical:
				f.write(str(elem)+',')
			f.write('\n')



'''
code to calculate sensitivity at given specificity value
'''

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


'''load METABRIC CNV dataset'''
dataset_cnv = numpy.loadtxt("A:/Project/Stacked Ensemble/Data/METABRIC_cnv_1980.txt", delimiter="\t")# Change the path to your local system
'''split into input (X) and output (Y) variables'''
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]


'''load TCGA CNV dataset'''
#X_cnv = numpy.loadtxt("A:/Project/Attention/Data/TCGA/tcga_cnv_final.csv", delimiter=",")# Change the path to your local system
#Y_cnv = numpy.loadtxt("A:/Project/Attention/Data/TCGA/5yearCutOff.txt")



# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores = []
i=1

for train_index, test_index in kfold.split(X_cnv, Y_cnv):
    print("*********************************************************************",i,"th Fold ********************************************************************")
    i=i+1
	#Spliting the clinical data set into training and testing
    x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
    y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index] 	
    x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
    x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)

    # first cnv CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=0.1)
    main_input_cnv = Input(shape=(200,1),name='Input')
    
    conv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv1D_cnv1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_cnv)
    gatedAtnConv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv1)
    gatedAtnConv_cnv1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv1)
    mult_1_1 = multiply([gatedAtnConv_cnv1,conv_cnv1])
    mult_1_1_1 = multiply([gatedAtnConv_cnv1_1,conv_cnv1])
    pooled_cnv1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_cnv1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    conv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=2,padding='same',name='Conv1D_cnv2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_cnv)
    gatedAtnConv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv2)
    gatedAtnConv_cnv2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv2)
    mult_2_2 = multiply([gatedAtnConv_cnv2,conv_cnv2])
    mult_2_2_2 = multiply([gatedAtnConv_cnv2_2,conv_cnv2])
    pooled_cnv2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_cnv2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    
    merged = concatenate([pooled_cnv1, pooled_cnv2,pooled_cnv1_1, pooled_cnv2_2],name='merge',axis=1)
    flat_cnv = Flatten(name='Flatten')(merged)
    
    dense_cnv = Dense(150,name='dense_cnv',activation='tanh',activity_regularizer=l2(0.01))(flat_cnv)
    

    

    drop_final1 = Dropout(rate = 0.5)(dense_cnv)
    dense_final2 = Dense(100,name='dense_final2',activation='tanh',activity_regularizer=l2(0.01))(drop_final1)
    dense_final3 = Dense(50,name='dense_final3',activation='tanh',activity_regularizer=l2(0.01))(dense_final2)
    output = Dense(1,activation='sigmoid')(dense_final3)    
    model = Model(inputs=main_input_cnv, outputs=output)
    plot_model(model, to_file='A:/Project/Attention/cnv_gated_attention.png') # Change the path to your local system


    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
    model.fit(x_train1, y_train1, epochs=epochs, batch_size=8,validation_data=(x_val1,y_val1))	


    scores = model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    Sn_cvscores.append(scores[1] * 100)
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))

X_cnv = numpy.expand_dims(X_cnv, axis=2)


intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final3").output)

intermediate_output = intermediate_layer_model.predict(X_cnv)
shows_result(path,intermediate_output)


y_pred = model.predict(X_cnv)
fpr, tpr, thresholds = roc_curve(Y_cnv, y_pred,pos_label=1)

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)
    

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'SiGaAtCNN-CNV = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

