# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:49:17 2020

@author: Aryan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:51:17 2020

@author: Aryan
"""
import keras
import os
import keras.backend as K
from keras.layers import Input,Dropout, Flatten,Dense,MaxPooling1D
from keras.layers import multiply, concatenate
#from keras.layers.merge import concatenate
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy
from keras import initializers,regularizers
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt

num_of_filters=25
epochs_cnn = 20
epochs_agcnn_cln=50
epochs_agcnn_exp=40
epochs_agcnn_cnv=40
Sp_value = 0.95
kfold_value = 10
path = "A:/Project/Attention/Data"
filecln = "METABRIC_clinical_1980.txt"
fileexp = "METABRIC_gene_exp_1980.txt"
filecnv = "METABRIC_cnv_1980.txt"
'''filecln = "METABRIC_clinical_1916_uncensored.txt"
fileexp = "METABRIC_gene_exp_1916_uncensored.txt"
filecnv = "METABRIC_cnv_1916_uncensored.txt"'''

pathfile = 'A:/Project/Attention/gatedAtnClnExpCnv.csv'
def shows_result(pathfile,arr):
	with open(pathfile, 'w') as f: # Change the path to your local system
		for item_clinical in arr:
			for elem in item_clinical:
				f.write(str(elem)+',')
			f.write('\n')
        
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    Sn=(true_positives / (possible_positives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sn 
        #Sn=sess.run(Sn)		# from tensor  to numpy
    return Sn

def specificity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    Sp= (true_negatives / (possible_negatives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sp 
       # Sp=sess.run(Sp)		# from tensor  to numpy
    return Sp

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


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# fix random seed for reproducibility
numpy.random.seed(1)

# load dataset
dataset_clinical = numpy.loadtxt(os.path.join(path, filecln), delimiter="\t") # Change the path to your local system
dataset_exp = numpy.loadtxt(os.path.join(path, fileexp), delimiter="\t")# Change the path to your local system
dataset_cnv = numpy.loadtxt(os.path.join(path, filecnv), delimiter="\t")# Change the path to your local system

# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]
# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]
# split into input (X) and output (Y) variables
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]

print('*********************************Training the Clinical CNN *****************************************')
# kfold_value fold cross validation
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)
acc_clinical = []
Pr_clinical = []
Sn_clinical = []
Mcc_clinical = []
i=1  
for train_index, test_index in kfold.split(X_clinical, Y_clinical):
	print(i,"th Fold *****************************************")
	i=i+1
	x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
	y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
	x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
	x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)
	# first Clinical CNN Model
	init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(25,1),name='Input')
	conv1 = Conv1D(filters=25,kernel_size=15,strides=2,activation='tanh',padding='same',name='Conv1D',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten(name='Flatten')(conv1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(flat1)
	output = Dense(1, activation='sigmoid',name='output',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense1)
	clinical_model = Model(inputs=main_input1, outputs=output)
	clinical_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	clinical_model.fit(x_train, y_train, epochs=epochs_cnn, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = clinical_model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	print("%s: %.2f%%" % (clinical_model.metrics_names[1], clinical_scores[1]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[2], clinical_scores[2]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[3], clinical_scores[3]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[4], clinical_scores[4]*100))
	acc_clinical.append(clinical_scores[1] * 100)
	Pr_clinical.append(clinical_scores[2] * 100)
	Sn_clinical.append(clinical_scores[3] * 100)
	Mcc_clinical.append(clinical_scores[4] * 100)
	intermediate_layer_clinical = Model(inputs=main_input1,outputs=dense1)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_clinical), numpy.std(acc_clinical)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_clinical), numpy.std(Pr_clinical)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_clinical), numpy.std(Sn_clinical)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_clinical), numpy.std(Mcc_clinical)))

print('***************************************************** Training the CNA CNN *****************************************************')
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)
acc_cnv = []
Pr_cnv = []
Sn_cnv = []
Mcc_cnv = []
i=1
for train_index, test_index in kfold.split(X_cnv, Y_cnv):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
	y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index]
	x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
	x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)
	
	# first CNV Model
	#init=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(200,1))
	conv1 = Conv1D(filters=4,kernel_size=15,strides=2,activation='tanh',padding='same',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten()(conv1)
	dropout1 = Dropout(0.50)(flat1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(dropout1)
	dropout2 = Dropout(0.25,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	cnv_model =	Model(inputs=main_input1, outputs=output)
	cnv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	cnv_model.fit(x_train_cnv, y_train_cnv, epochs=epochs_cnn,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = cnv_model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (cnv_model.metrics_names[1], cnv_scores[1]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[2], cnv_scores[2]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[3], cnv_scores[3]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[4], cnv_scores[4]*100))
	acc_cnv.append(cnv_scores[1] * 100)
	Pr_cnv.append(cnv_scores[2] * 100)
	Sn_cnv.append(cnv_scores[3] * 100)
	Mcc_cnv.append(cnv_scores[4] * 100)
	intermediate_layer_cnv = Model(inputs=main_input1,outputs=dropout2)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_cnv), numpy.std(acc_cnv)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_cnv), numpy.std(Pr_cnv)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cnv), numpy.std(Sn_cnv)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_cnv), numpy.std(Mcc_cnv)))



print('************************************************** Training the Expr CNN *********************************************')
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)
acc_exp = []
Pr_exp = []
Sn_exp = []
Mcc_exp = []
i=1
for train_index, test_index in kfold.split(X_exp, Y_exp):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
	y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 
	x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
	x_test_exp = numpy.expand_dims(x_test_exp, axis=2)
	
	# first CNN EXP Model
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(400,1))
	conv1 = Conv1D(filters=4,kernel_size=15,strides=2,activation='tanh',padding='same',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten()(conv1)
	dropout1 = Dropout(0.50)(flat1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(dropout1)
	dropout2 = Dropout(0.25,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	exp_model =	Model(inputs=main_input1, outputs=output)
	exp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	exp_model.fit(x_train, y_train, epochs=epochs_cnn, batch_size=8,verbose=2,validation_data=(x_val,y_val))
	exp_scores = exp_model.evaluate(x_test_exp, y_test_exp,verbose=2)
	print("%s: %.2f%%" % (exp_model.metrics_names[1], exp_scores[1]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[2], exp_scores[2]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[3], exp_scores[3]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[4], exp_scores[4]*100))
	acc_exp.append(exp_scores[1] * 100)
	Pr_exp.append(exp_scores[2] * 100)
	Sn_exp.append(exp_scores[3] * 100)
	Mcc_exp.append(exp_scores[4] * 100)
	intermediate_layer_exp = Model(inputs=main_input1,outputs=dropout2)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_exp), numpy.std(acc_exp)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_exp), numpy.std(Pr_exp)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_exp), numpy.std(Sn_exp)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_exp), numpy.std(Mcc_exp)))

    
print('*********************************Training the Clinical AGCNN *****************************************')
# kfold_value fold cross validation
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)
Sn_cln_cvscores = []
i=1  
for train_index, test_index in kfold.split(X_clinical, Y_clinical):
    print(i,"th Fold *****************************************")
    i=i+1
    x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
    y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
    x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
    x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)
    # first Clinical CNN Model    
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=0.1)
    main_input_clinical = Input(shape=(25,1),name='Inputcln')
    
    conv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=2,padding='same',name='Conv1D_clinical1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    gatedAtnConv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cln1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    gatedAtnConv_clinical1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cln1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    mult_1_1 = multiply([gatedAtnConv_clinical1,conv_clinical1])
    mult_1_1_1 = multiply([gatedAtnConv_clinical1_1,conv_clinical1])
    pooled_clinical1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_clinical1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    conv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv1D_clinical2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    gatedAtnConv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cln1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    gatedAtnConv_clinical2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cln1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    mult_2_2 = multiply([gatedAtnConv_clinical2,conv_clinical2])
    mult_2_2_2 = multiply([gatedAtnConv_clinical2_2,conv_clinical2])
    pooled_clinical2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_clinical2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    merged_clinical = concatenate([pooled_clinical1, pooled_clinical2,pooled_clinical1_1, pooled_clinical2_2],name='mergecln',axis=1)
    flat_clinical = Flatten(name='Flattencln')(merged_clinical)
    
    dense_clinical = Dense(150,name='dense_clinical',activation='tanh',activity_regularizer=l2(0.001))(flat_clinical)
    drop_clinical = Dropout(rate = 0.25)(dense_clinical)
    dense_clinical = Dense(100,name='dense_clinical2',activation='tanh',activity_regularizer=l2(0.001))(drop_clinical)
    dense_clinical = Dense(50,name='dense_clinical3',activation='tanh',activity_regularizer=l2(0.001))(dense_clinical)
    output_clinical = Dense(1,activation='sigmoid')(dense_clinical)    
    model_clinical = Model(inputs=main_input_clinical, outputs=output_clinical)
    plot_model(model_clinical, to_file='A:/Project/Attention/clinical_gated_attention.png') # Change the path to your local system

    model_clinical.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
    model_clinical.fit(x_train1, y_train1, epochs=epochs_agcnn_cln, batch_size=8,validation_data=(x_val1,y_val1))	


    scores_clinical = model_clinical.evaluate(x_test_clinical, y_test_clinical,verbose=2)
    print("%s: %.2f%%" % (model_clinical.metrics_names[1], scores_clinical[1]*100))
    Sn_cln_cvscores.append(scores_clinical[1] * 100)
    
    intermediate_layer_model_clinical = Model(inputs=model_clinical.input, outputs=model_clinical.get_layer("dense_clinical").output)
    
    #shows_result(path,intermediate_output_clinical)
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cln_cvscores), numpy.std(Sn_cln_cvscores)))    
    
print('***************************************************** Training the EXP AGCNN *****************************************************')
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)   
Sn_exp_cvscores =[]
i=1
for train_index, test_index in kfold.split(X_exp, Y_exp):
    print(i,'th Fold *******************************')
    i=i+1
	#Spliting the data set into training and testing
    x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
    y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 
    x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
    x_test_exp = numpy.expand_dims(x_test_exp, axis=2)
 
    # first exp CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=1)
    main_input_exp = Input(shape=(400,1),name='Inputexp')
    
    conv_exp1 = Conv1D(filters=num_of_filters,kernel_size=2,strides=1,padding='same',name='Conv1D_exp1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_exp1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    gatedAtnConv_exp1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_exp1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp1)
    mult_1_1 = multiply([gatedAtnConv_exp1,conv_exp1])
    mult_1_1_1 = multiply([gatedAtnConv_exp1_1,conv_exp1])
    pooled_exp1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_exp1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    
    conv_exp2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='Conv1D_exp2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_exp)
    gatedAtnConv_exp2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_exp1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    gatedAtnConv_exp2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_exp1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_exp2)
    mult_2_2 = multiply([gatedAtnConv_exp2,conv_exp2])
    mult_2_2_2 = multiply([gatedAtnConv_exp2_2,conv_exp2])
    pooled_exp2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_exp2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    merged_exp = concatenate([pooled_exp1,pooled_exp1_1,pooled_exp2,pooled_exp2_2],name='merge',axis=1)
    flat_exp = Flatten(name='Flattenexp')(merged_exp)
    dense_exp = Dense(150,name='dense_exp',activation='tanh',activity_regularizer=l2(0.01))(flat_exp)
    drop_exp = Dropout(rate = 0.5)(dense_exp)
    dense_exp = Dense(100,name='dense_exp2',activation='tanh',activity_regularizer=l2(0.01))(drop_exp)
    dense_exp = Dense(50,name='dense_exp3',activation='tanh',activity_regularizer=l2(0.01))(dense_exp)
    output_exp = Dense(1,activation='sigmoid')(dense_exp)
    model_exp = Model(inputs=main_input_exp, outputs=output_exp)
    plot_model(model_exp, to_file='A:/Project/Attention/exp_gated_attention.png') # Change the path to your local system

    model_exp.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)
    model_exp.fit(x_train1, y_train1, epochs=epochs_agcnn_exp, batch_size=8,validation_data=(x_val1,y_val1))	
    scores_exp = model_exp.evaluate(x_test_exp, y_test_exp,verbose=2)
    print("%s: %.2f%%" % (model_exp.metrics_names[1], scores_exp[1]*100))
    Sn_exp_cvscores.append(scores_exp[1] * 100)
    
    intermediate_layer_model_exp = Model(inputs=model_exp.input, outputs=model_exp.get_layer("dense_exp").output)
    
    #shows_result(path,intermediate_output_exp)    
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_exp_cvscores), numpy.std(Sn_exp_cvscores)))   
    
print('***************************************************** Training the CNA AGCNN *****************************************************')
kfold = StratifiedKFold(n_splits=kfold_value, shuffle=False, random_state=1)
Sn_cnv_cvscores = []
i=1
for train_index, test_index in kfold.split(X_cnv, Y_cnv):
    print(i,'th Fold *******************************')
    i=i+1
	#Spliting the data set into training and testing
    x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
    y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index]
    x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
    x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)
	
	# first CNV Model
	#init=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    bias_init =keras.initializers.Constant(value=0.1)
    main_input_cnv = Input(shape=(200,1),name='Inputcnv')
    
    conv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv1D_cnv1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_cnv)
    gatedAtnConv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cnv1D1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv1)
    gatedAtnConv_cnv1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cnv1D1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv1)
    mult_1_1 = multiply([gatedAtnConv_cnv1,conv_cnv1])
    mult_1_1_1 = multiply([gatedAtnConv_cnv1_1,conv_cnv1])
    pooled_cnv1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_cnv1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    conv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=2,padding='same',name='Conv1D_cnv2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(main_input_cnv)
    gatedAtnConv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cnv1D2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv2)
    gatedAtnConv_cnv2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cnv1D2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.01))(conv_cnv2)
    mult_2_2 = multiply([gatedAtnConv_cnv2,conv_cnv2])
    mult_2_2_2 = multiply([gatedAtnConv_cnv2_2,conv_cnv2])
    pooled_cnv2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_cnv2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    merged_cnv = concatenate([pooled_cnv1, pooled_cnv2,pooled_cnv1_1, pooled_cnv2_2],name='merge_cnv',axis=1)
    flat_cnv = Flatten(name='Flatten_cnv')(merged_cnv)
    
    dense_cnv = Dense(150,name='dense_cnv',activation='tanh',activity_regularizer=l2(0.01))(flat_cnv)
    drop_cnv = Dropout(rate = 0.5)(dense_cnv)
    dense_cnv = Dense(100,name='dense_cnv2',activation='tanh',activity_regularizer=l2(0.01))(drop_cnv)
    dense_cnv = Dense(50,name='dense_cnv3',activation='tanh',activity_regularizer=l2(0.01))(dense_cnv)
    output_cnv = Dense(1,activation='sigmoid')(dense_cnv)    
    model_cnv = Model(inputs=main_input_cnv, outputs=output_cnv)
    plot_model(model_cnv, to_file='A:/Project/Attention/cnv_gated_attention.png') # Change the path to your local system

    model_cnv.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
    model_cnv.fit(x_train1, y_train1, epochs=epochs_agcnn_cnv, batch_size=8,validation_data=(x_val1,y_val1))	

    scores_cnv = model_cnv.evaluate(x_test_cnv, y_test_cnv,verbose=2)
    print("%s: %.2f%%" % (model_cnv.metrics_names[1], scores_cnv[1]*100))
    Sn_cnv_cvscores.append(scores_cnv[1] * 100)
    intermediate_layer_model_cnv = Model(inputs=model_cnv.input, outputs=model_cnv.get_layer("dense_cnv").output)
    
    
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cnv_cvscores), numpy.std(Sn_cnv_cvscores)))

X_clinical = numpy.expand_dims(X_clinical, axis=2)
X_exp=numpy.expand_dims(X_exp, axis=2)
X_cnv = numpy.expand_dims(X_cnv, axis=2)

intermediate_output_cnv = intermediate_layer_model_cnv.predict(X_cnv)
intermediate_output_exp = intermediate_layer_model_exp.predict(X_exp)
intermediate_output_clinical = intermediate_layer_model_clinical.predict(X_clinical)
stacked_feature=numpy.concatenate((intermediate_output_clinical,intermediate_output_cnv,intermediate_output_exp,Y_clinical[:,None]),axis=1)
shows_result(pathfile,stacked_feature)


#Plotting
X_train_clinical, X_test_clinical, y_train_clinical, y_test_clinical = train_test_split(X_clinical, Y_clinical, test_size=0.2,stratify=Y_clinical)
X_train_cnv, X_test_cnv, y_train_cnv, y_test_cnv = train_test_split(X_cnv, Y_cnv, test_size=0.2,stratify=Y_cnv)
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y_exp, test_size=0.2,stratify=Y_exp)
'''X_test_clinical=numpy.expand_dims(X_test_clinical, axis=2)
X_test_cnv=numpy.expand_dims(X_test_cnv, axis=2)
X_test_exp=numpy.expand_dims(X_test_exp, axis=2)
X_cnv = numpy.expand_dims(X_cnv, axis=2)
X_exp=numpy.expand_dims(X_exp, axis=2)
X_clinical = numpy.expand_dims(X_clinical, axis=2)'''
pred_clinical = clinical_model.predict(X_test_clinical)
pred_cnv = cnv_model.predict(X_test_cnv)
pred_exp = exp_model.predict(X_test_exp)
fpr_clinical, tpr_clinical, threshold_clinical = roc_curve(y_test_clinical, pred_clinical,pos_label=1)
fpr_cnv, tpr_cnv, threshold_cnv = roc_curve(y_test_cnv, pred_cnv,pos_label=1)
fpr_exp, tpr_exp, threshold_exp = roc_curve(y_test_exp, pred_exp,pos_label=1)

pred_agcnnclinical = model_clinical.predict(X_test_clinical)
pred_agcnncnv = model_cnv.predict(X_test_cnv)
pred_agcnnexp = model_exp.predict(X_test_exp)
fpr_agcnnclinical, tpr_agcnnclinical, threshold_agcnnclinical = roc_curve(y_test_clinical, pred_agcnnclinical,pos_label=1)
fpr_agcnncnv, tpr_agcnncnv, threshold_agcnncnv = roc_curve(y_test_cnv, pred_agcnncnv,pos_label=1)
fpr_agcnnexp, tpr_agcnnexp, threshold_agcnnexp = roc_curve(y_test_exp, pred_agcnnexp,pos_label=1)

roc_auc_clinical = auc(fpr_clinical, tpr_clinical)
roc_auc_cnv = auc(fpr_cnv, tpr_cnv)
roc_auc_exp = auc(fpr_exp, tpr_exp)

roc_auc_agcnnclinical = auc(fpr_agcnnclinical, tpr_agcnnclinical)
roc_auc_agcnncnv = auc(fpr_agcnncnv, tpr_agcnncnv)
roc_auc_agcnnexp = auc(fpr_agcnnexp, tpr_agcnnexp)

plt.title('Receiver Operating Characteristic')




plt.plot(fpr_agcnnclinical, tpr_agcnnclinical, 'y', label = 'AGCNN-CLN = %0.3f' % roc_auc_agcnnclinical)
plt.plot(fpr_clinical, tpr_clinical, 'r', label = 'CNN-CLN = %0.3f' % roc_auc_clinical)

plt.plot(fpr_agcnncnv, tpr_agcnncnv, 'o', label = 'AGCNN-CNA = %0.3f' % roc_auc_agcnncnv)
plt.plot(fpr_cnv, tpr_cnv, 'b', label = 'CNN-CNA = %0.3f' % roc_auc_cnv)

plt.plot(fpr_agcnnexp, tpr_agcnnexp, 'c', label = 'AGCNN-EXPR = %0.3f' % roc_auc_agcnnexp)
plt.plot(fpr_exp, tpr_exp, 'g', label = 'CNN-EXPR = %0.3f' % roc_auc_exp)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate(Sn)')
plt.xlabel('Flase Positive Rate(1-Sp)')
plt.show()


'''def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)'''
	
    