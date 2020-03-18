from keras.layers import Input,Dropout, Flatten,Dense
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy,math
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras import initializers,regularizers,optimizers
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix 
import keras.backend as K
import numpy as np
import os

epochs = 25
Sp_value = 0.95
path = "A:/Project/Attention/Data/TCGA/"
filecln = "tcga_cln_final.csv"
fileexp = "tcga_exp_final.csv"
filecnv = "tcga_cnv_final.csv"
label = "5yearCutOff.txt"
stacked_metadata="TCGA_stacked_metadata.csv"


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

# load Clinical dataset
X_clinical = numpy.loadtxt(os.path.join(path, filecln), delimiter=",")#change the path to your local system	
X_cnv = numpy.loadtxt(os.path.join(path, filecnv), delimiter=",") #change the path to your local system
X_exp = numpy.loadtxt(os.path.join(path, fileexp), delimiter=",")#change the path to your local system

Y_clinical = numpy.loadtxt(os.path.join(path, label))
Y_cnv = numpy.loadtxt(os.path.join(path, label))
Y_exp = numpy.loadtxt(os.path.join(path, label))


print('Training the Clinical CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
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
	main_input1 = Input(shape=(11,1),name='Input')
	conv1 = Conv1D(filters=10,kernel_size=15,strides=2,activation='tanh',padding='same',name='Conv1D',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten(name='Flatten')(conv1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(flat1)
	output = Dense(1, activation='sigmoid',name='output',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense1)
	clinical_model = Model(inputs=main_input1, outputs=output)
	clinical_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[sensitivity_at_specificity(0.95)])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	clinical_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = clinical_model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	
	print("%s: %.2f%%" % (clinical_model.metrics_names[1], clinical_scores[1]*100))
	
	
	Sn_clinical.append(clinical_scores[1] * 100)
	
	intermediate_layer_clinical = Model(inputs=main_input1,outputs=dense1)

print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_clinical), numpy.std(Sn_clinical)))


print('Training the CNA CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
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
	dropout2 = Dropout(0.10,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	cnv_model =	Model(inputs=main_input1, outputs=output)
	cnv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[sensitivity_at_specificity(0.95)])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	cnv_model.fit(x_train_cnv, y_train_cnv, epochs=epochs,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = cnv_model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (cnv_model.metrics_names[1], cnv_scores[1]*100))
	
	
	Sn_cnv.append(cnv_scores[1] * 100)
	
	intermediate_layer_cnv = Model(inputs=main_input1,outputs=dropout2)

print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cnv), numpy.std(Sn_cnv)))




print('Training the Expr CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
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
	dropout2 = Dropout(0.10,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	exp_model =	Model(inputs=main_input1, outputs=output)
	exp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[sensitivity_at_specificity(0.95)])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	exp_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))
	exp_scores = exp_model.evaluate(x_test_exp, y_test_exp,verbose=2)
	
	print("%s: %.2f%%" % (exp_model.metrics_names[1], exp_scores[1]*100))
	
	
	Sn_exp.append(exp_scores[1] * 100)
	
	intermediate_layer_exp = Model(inputs=main_input1,outputs=dropout2)

print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_exp), numpy.std(Sn_exp)))




X_clinical_ = numpy.expand_dims(X_clinical, axis=2)
# for extracting final layer features 
#y_pred_ =  clinical_model.predict(X_clinical_)
# for extracting one layer before final layer features
y_pred_clinical = intermediate_layer_clinical.predict(X_clinical_)

X_cnv_ = numpy.expand_dims(X_cnv, axis=2)
# for extracting final layer features 
#y_pred_ =  cnv_model.predict(X_cnv_)
# for extracting one layer before final layer features
y_pred_cnv = intermediate_layer_cnv.predict(X_cnv_)

X_exp_ = numpy.expand_dims(X_exp, axis=2)
# for extracting final layer features 
#y_pred_ =  exp_model.predict(X_exp_)
# for extracting one layer before final layer features
y_pred_exp = intermediate_layer_exp.predict(X_exp_)

stacked_feature=numpy.concatenate((y_pred_clinical,y_pred_cnv,y_pred_exp,Y_clinical[:,None]),axis=1)
with open(os.path.join(path, stacked_metadata), 'w') as f:    #change the path to your local system
	for item_clinical in stacked_feature:
		for elem in item_clinical:
			f.write(str(elem)+'\t')
		f.write('\n')

#Plotting
X_train_clinical, X_test_clinical, y_train_clinical, y_test_clinical = train_test_split(X_clinical, Y_clinical, test_size=0.2,stratify=Y_clinical)
X_train_cnv, X_test_cnv, y_train_cnv, y_test_cnv = train_test_split(X_cnv, Y_cnv, test_size=0.2,stratify=Y_cnv)
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y_exp, test_size=0.2,stratify=Y_exp)

X_test_clinical=numpy.expand_dims(X_test_clinical, axis=2)
X_test_cnv=numpy.expand_dims(X_test_cnv, axis=2)
X_test_exp=numpy.expand_dims(X_test_exp, axis=2)
pred_clinical = clinical_model.predict(X_test_clinical)
pred_cnv = cnv_model.predict(X_test_cnv)
pred_exp = exp_model.predict(X_test_exp)
fpr_clinical, tpr_clinical, threshold_clinical = roc_curve(y_test_clinical, pred_clinical,pos_label=1)
fpr_cnv, tpr_cnv, threshold_cnv = roc_curve(y_test_cnv, pred_cnv,pos_label=1)
fpr_exp, tpr_exp, threshold_exp = roc_curve(y_test_exp, pred_exp,pos_label=1)

def evaluate_threshold_clinical(threshold):
    print("Clinical")
    print('Sensitivity:', tpr_clinical[threshold_clinical > threshold][-1])
    print('Specificity:', 1 - fpr_clinical[threshold_clinical > threshold][-1])
    
def evaluate_threshold_cnv(threshold):
    print("CNA")
    print('Sensitivity:', tpr_cnv[threshold_cnv > threshold][-1])
    print('Specificity:', 1 - fpr_cnv[threshold_cnv > threshold][-1])
    
def evaluate_threshold_exp(threshold):
    print("EXPR")
    print('Sensitivity:', tpr_exp[threshold_exp > threshold][-1])
    print('Specificity:', 1 - fpr_exp[threshold_exp > threshold][-1])


for threshold in numpy.arange(0,1,0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold_clinical(threshold)
    evaluate_threshold_cnv(threshold)
    evaluate_threshold_exp(threshold)
    
roc_auc_clinical = auc(fpr_clinical, tpr_clinical)
roc_auc_cnv = auc(fpr_cnv, tpr_cnv)
roc_auc_exp = auc(fpr_exp, tpr_exp)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_clinical, tpr_clinical, 'r', label = 'CNN-Clinical = %0.2f' % roc_auc_clinical)
plt.plot(fpr_cnv, tpr_cnv, 'b', label = 'CNN-CNA = %0.2f' % roc_auc_cnv)
plt.plot(fpr_exp, tpr_exp, 'g', label = 'CNN-Expr = %0.2f' % roc_auc_exp)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate(Sn)')
plt.xlabel('Flase Positive Rate(1-Sp)')
plt.show()




