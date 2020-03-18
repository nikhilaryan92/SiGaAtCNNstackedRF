# SiGaAtCNNstackedRF

# Multi-Modal Advanced Deep Learning Architectures for Breast Cancer Survival Prediction

# References

Our manuscipt titled with "Multi-Modal Advanced Deep Learning Architectures for Breast Cancer Survival Prediction" has been submitted to Scientific reports journal by Nature Research.

# Requirements
[python 3.6](https://www.python.org/downloads/)


[TensorFilow 1.12](https://www.tensorflow.org/install/)

[keras 2.2.4](https://pypi.org/project/Keras/)


[scikit-learn 0.20.0](http://scikit-learn.org/stable/)


[matplotlib 3.0.1](https://matplotlib.org/users/installing.html)



# Usage
SiGaAtCNN_cln.py

SiGaAtCNN_cnv.py

SiGaAtCNN_expr.py

SiGaAtCNN_vs_CNN.py

RF_All.py

ttest.py

# Process to execute the SiGaAtCNN + Input STACKED RF architecture.

=>  Run SiGaAtCNN_cln.py, SiGaAtCNN_cnv.py, SiGaAtCNN_expr.py for training individual SiGaAtCNNs for clinical, CNA and gene-expression data.

=>  After successfull run you will get the hidden features in three different csv files : gatedAtnClnOutput.csv, gatedAtnCnvOutput.csv and gatedAtnExpOutput.csv

=> Combine all the hidden features of different modalities along with their respective input feaures to form stacked features : gatedAtnAll_Input.csv

=>  run AdaboostRF.py and pass the stacked feature(gatedAtnAll_Input.csv) as input to get the final prediction output.

=>  Once final prediction has been made use ttest.py to perform statistical significance test.

# For comparative study of uni-modal SiGaAtCNNs and simple CNNs

=> Run the SiGaAtCNN_vs_CNN.py to plot the ROC curves of sigmoid gated attention CNN vs simple CNN.

# For comparative study of our proposed approach with some other methods

=> Run the RF_All.py to plot ROC curves of SiGaAtCNN + Input STACKED RF, SiGaAtCNN STACKED RF, SiGaAtCNN Bi-Attention, SiGaAtCNN Bi-Attention STACKED RF






