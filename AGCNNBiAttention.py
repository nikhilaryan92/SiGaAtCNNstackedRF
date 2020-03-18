# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:20:35 2020

@author: Aryan
"""

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

num_of_filters = 10
epochs = 40
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []
path = 'A:/Project/Attention/GatedBiAttentionOutput.csv'
path1 = 'A:/Project/Attention/GatedBiAttentionOutput1.csv'
def shows_result(path,arr):
	with open(path, 'w') as f: # Change the path to your local system
		for item_clinical in arr:
			for elem in item_clinical:
				f.write(str(elem)+',')
			f.write('\n')





def bi_modal_attention(x, y):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
        
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
       
    return {a1, a2}
        
    '''
     
    m1 = dot([x, y], axes=[2, 2])
    n1 = Activation('softmax')(m1)
    
    o1 = dot([ n1, y],axes=[2,1])
    
    a1 = multiply([o1, x])
    
    m2 = dot([y, x], axes=[2, 2])
    
    n2 = Activation('softmax')(m2)
    
    o2 = dot([n2, x],axes=[2,1])
    
    a2 = multiply([o2, y])
    return concatenate([a1, a2],axis=1)

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

# load Clinical dataset
dataset_clinical = numpy.loadtxt("A:/Project/Stacked Ensemble/Data/METABRIC_clinical_1980.txt", delimiter="\t") # Change the path to your local system
dataset_exp = numpy.loadtxt("A:/Project/Stacked Ensemble/Data/METABRIC_gene_exp_1980.txt", delimiter="\t")# Change the path to your local system
dataset_cnv = numpy.loadtxt("A:/Project/Stacked Ensemble/Data/METABRIC_cnv_1980.txt", delimiter="\t")# Change the path to your local system


# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]



# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores = []
i=1

for train_index, test_index in kfold.split(X_clinical, Y_clinical):
    print(i,"th Fold *****************************************")
    i=i+1
	#Spliting the clinical data set into training and testing
    x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
    y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
    x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
    x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)
    #Spliting the exp data set into training and testing
    x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
    y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 
    x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
    x_test_exp = numpy.expand_dims(x_test_exp, axis=2)
	#Spliting the cnv data set into training and testing
    x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
    y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index]
    x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
    x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)


    # first Clinical CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=0.5)
    main_input_clinical = Input(shape=(25,1),name='Input_clinical')
    
    conv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=2,padding='same',name='Conv_clinical1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    gatedAtnConv_clinical1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_CLN1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    gatedAtnConv_clinical1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_CLN1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical1)
    mult_1_1 = multiply([gatedAtnConv_clinical1,conv_clinical1])
    mult_1_1_1 = multiply([gatedAtnConv_clinical1_1,conv_clinical1])
    pooled_clinical1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_clinical1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    conv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv_clinical2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(main_input_clinical)
    gatedAtnConv_clinical2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_CLN2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    gatedAtnConv_clinical2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_CLN2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.001))(conv_clinical2)
    mult_2_2 = multiply([gatedAtnConv_clinical2,conv_clinical2])
    mult_2_2_2 = multiply([gatedAtnConv_clinical2_2,conv_clinical2])
    pooled_clinical2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_clinical2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2) 
    
    merged_clinical = concatenate([pooled_clinical1, pooled_clinical2,pooled_clinical1_1, pooled_clinical2_2],name='merge_clinical',axis=1)
    '''flat_clinical = Flatten(name='Flatten_clinical')(merged_clinical)
    
    dense_clinical = Dense(150,name='dense_clinical1',activation='tanh',activity_regularizer=l2(0.001))(flat_clinical)
    dense_clinical = Dropout(rate = 0.25)(dense_clinical)
    dense_clinical = Dense(100,name='dense_clinical2',activation='tanh',activity_regularizer=l2(0.001))(dense_clinical)
    dense_clinical = Dense(50,name='dense_clinical3',activation='tanh',activity_regularizer=l2(0.001))(dense_clinical)'''
    

    
    
     # first exp CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=0.5)
    main_input_exp = Input(shape=(400,1),name='Input_exp')
    
    conv_exp1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='Conv_exp1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(main_input_exp)
    gatedAtnConv_exp1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_exp1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_exp1)
    gatedAtnConv_exp1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_exp1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_exp1)
    mult_1_1 = multiply([gatedAtnConv_exp1,conv_exp1])
    mult_1_1_1 = multiply([gatedAtnConv_exp1_1,conv_exp1])
    pooled_exp1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_exp1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    
    conv_exp2 = Conv1D(filters=num_of_filters,kernel_size=2,strides=1,padding='same',name='Conv_exp2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(main_input_exp)
    gatedAtnConv_exp2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_exp2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_exp2)
    gatedAtnConv_exp2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_exp2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_exp2)
    mult_2_2 = multiply([gatedAtnConv_exp2,conv_exp2])
    mult_2_2_2 = multiply([gatedAtnConv_exp2_2,conv_exp2])
    pooled_exp2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_exp2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
   
   
    merged_exp = concatenate([pooled_exp1,pooled_exp2,pooled_exp1_1,pooled_exp2_2],name='merge_exp',axis=1)
    '''flat_exp = Flatten(name='Flatten_exp')(merged_exp)
    dense_exp = Dense(150,name='dense_exp1',activation='tanh',activity_regularizer=l2(0.4))(flat_exp)
    dense_exp = Dropout(rate = 0.5)(dense_exp)
    dense_exp = Dense(100,name='dense_exp2',activation='tanh',activity_regularizer=l2(0.4))(dense_exp)
    dense_exp = Dense(50,name='dense_exp3',activation='tanh',activity_regularizer=l2(0.4))(dense_exp)'''
    
    
     # first cnv CNN Model***********************************************************
    #init =initializers.glorot_normal(seed=1)
    bias_init =keras.initializers.Constant(value=0.5)
    main_input_cnv = Input(shape=(200,1),name='Input_cnv')
    
    conv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=2,padding='same',name='Conv_cnv1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(main_input_cnv)
    gatedAtnConv_cnv1 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cnv1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_cnv1)
    gatedAtnConv_cnv1_1 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cnv1_1',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_cnv1)
    mult_1_1 = multiply([gatedAtnConv_cnv1,conv_cnv1])
    mult_1_1_1 = multiply([gatedAtnConv_cnv1_1,conv_cnv1])
    pooled_cnv1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1)
    pooled_cnv1_1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_1_1_1)
    
    conv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=2,strides=2,padding='same',name='Conv_cnv2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(main_input_cnv)
    gatedAtnConv_cnv2 = Conv1D(filters=num_of_filters,kernel_size=1,strides=1,padding='same',name='GatedConv_cnv2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_cnv2)
    gatedAtnConv_cnv2_2 = Conv1D(filters=num_of_filters,kernel_size=3,strides=1,padding='same',name='GatedConv_cnv2_2',activation='relu',kernel_initializer='glorot_uniform', bias_initializer=bias_init,activity_regularizer=l2(0.4))(conv_cnv2)
    mult_2_2 = multiply([gatedAtnConv_cnv2,conv_cnv2])
    mult_2_2_2 = multiply([gatedAtnConv_cnv2_2,conv_cnv2])
    pooled_cnv2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2)
    pooled_cnv2_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(mult_2_2_2)
    
    
    merged_cnv = concatenate([pooled_cnv1, pooled_cnv2,pooled_cnv1_1, pooled_cnv2_2],name='merge_cnv',axis=1)
    '''flat_cnv = Flatten(name='Flatten_cnv')(merged_cnv)
    
    dense_cnv = Dense(150,name='dense_cnv1',activation='tanh',activity_regularizer=l2(0.4))(flat_cnv)
    dense_cnv = Dropout(rate = 0.5)(dense_cnv)
    dense_cnv = Dense(100,name='dense_cnv2',activation='tanh',activity_regularizer=l2(0.4))(dense_cnv)
    dense_cnv = Dense(50,name='dense_cnv3',activation='tanh',activity_regularizer=l2(0.4))(dense_cnv)'''
    
    
    cln_cnv_att = bi_modal_attention(merged_clinical, merged_cnv)
    cnv_exp_att = bi_modal_attention(merged_cnv, merged_exp)
    exp_cln_att = bi_modal_attention(merged_exp, merged_clinical)
                
    merged = concatenate([cln_cnv_att, cnv_exp_att, exp_cln_att, merged_clinical, merged_cnv, merged_exp],name='merge',axis=1)
    
    flat_final = Flatten(name='flat_final')(merged)
    dense_final = Dense(150,name='dense_final1',activation='tanh',activity_regularizer=l2(0.4))(flat_final)
    drop_final1 = Dropout(rate = 0.5)(dense_final)
    dense_final2 = Dense(100,name='dense_final2',activity_regularizer=l2(0.001))(drop_final1)
    dense_final2 = Activation('relu')(dense_final2)
    dense_final3 = Dense(50,name='dense_final3',activity_regularizer=l2(0.001))(dense_final2)
    dense_final3 = Activation('relu')(dense_final3)
    output = Dense(1,activation='sigmoid')(dense_final3)
    model = Model(inputs=[main_input_clinical,main_input_exp,main_input_cnv], outputs=output)
    plot_model(model, to_file='A:/Project/Attention/gated_attention.png') # Change the path to your local system


	#model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])
    
    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
    x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)
    x_train3, x_val3, y_train3, y_val3 = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
    model.fit([x_train1,x_train2,x_train3], y_train1, epochs=epochs, batch_size=8,validation_data=([x_val1,x_val2,x_val3],y_val1))	
	
    
   
   
	#model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])


    scores = model.evaluate([x_test_clinical,x_test_exp,x_test_cnv], y_test_clinical,verbose=2)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))'''
    #acc_cvscores.append(scores[1] * 100)
	#Pr_cvscores.append(scores[2] * 100)
    Sn_cvscores.append(scores[1] * 100)
	#Mcc_cvscores.append(scores[4] * 100)
#print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_cvscores), numpy.std(acc_cvscores)))
#print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_cvscores), numpy.std(Pr_cvscores)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))
#print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_cvscores), numpy.std(Mcc_cvscores)))




X_clinical = numpy.expand_dims(X_clinical, axis=2)
X_exp = numpy.expand_dims(X_exp, axis=2)
X_cnv = numpy.expand_dims(X_cnv, axis=2)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense_final1").output)
intermediate_output = intermediate_layer_model.predict([X_clinical,X_exp,X_cnv])
shows_result(path,intermediate_output)

intermediate_layer_model1 = Model(inputs=model.input, outputs=model.get_layer("flat_final").output)
intermediate_output1 = intermediate_layer_model1.predict([X_clinical,X_exp,X_cnv])
shows_result(path1,intermediate_output1)

y_pred = model.predict([X_clinical,X_exp,X_cnv])
fpr, tpr, thresholds = roc_curve(Y_clinical, y_pred,pos_label=1)
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)
roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr, 'r', label = 'AGCNN Bi-Attention = %0.3f' %roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

