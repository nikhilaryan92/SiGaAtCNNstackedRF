from scipy import stats
import numpy as np
import os

path = "A:\Project\Attention\Ttest"
file1 = "ttest_metabric_acc.csv"
file2 = "ttest_metabric_AUC.csv"
file3 = "ttest_metabric_sn.csv"
file4 = "ttest_tcga_acc.csv"
file5 = "ttest_tcga_AUC.csv"
file6 = "ttest_tcga_sn.csv"
dataset1 = np.loadtxt(os.path.join(path, file1),delimiter=",")
dataset2 = np.loadtxt(os.path.join(path, file2),delimiter=",")
dataset3 = np.loadtxt(os.path.join(path, file3),delimiter=",")
dataset4 = np.loadtxt(os.path.join(path, file4),delimiter=",")
dataset5 = np.loadtxt(os.path.join(path, file5),delimiter=",")
dataset6 = np.loadtxt(os.path.join(path, file6),delimiter=",")


a1 = dataset1[:,0]
b1 = dataset1[:,1]

a2 = dataset2[:,0]
b2 = dataset2[:,1]

a3 = dataset3[:,0]
b3 = dataset3[:,1]

a4 = dataset4[:,0]
b4 = dataset4[:,1]

a5 = dataset5[:,0]
b5 = dataset5[:,1]

a6 = dataset6[:,0]
b6 = dataset6[:,1]


'''
t2, p2 = stats.ttest_ind(a1,b1)
print('METABRIC Acc')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.ttest_ind(a2,b2)
print('METABRIC AUC')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.ttest_ind(a3,b3)
print('METABRIC Sn')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.ttest_ind(a4,b4)
print('TCGA Acc')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.ttest_ind(a5,b5)
print('TCGA AUC')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.ttest_ind(a6,b6)
print('TCGA Sn')
print("t = " + str(t2))
print("p = " + str(2*p2))
'''


t2, p2 = stats.f_oneway(a1,b1)
print('METABRIC Acc')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.f_oneway(a2,b2)
print('METABRIC AUC')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.f_oneway(a3,b3)
print('METABRIC Sn')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.f_oneway(a4,b4)
print('TCGA Acc')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.f_oneway(a5,b5)
print('TCGA AUC')
print("t = " + str(t2))
print("p = " + str(2*p2))

t2, p2 = stats.f_oneway(a6,b6)
print('TCGA Sn')
print("t = " + str(t2))
print("p = " + str(2*p2))