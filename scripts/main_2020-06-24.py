for #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:58 2019

@author: dmason
"""

# Importing libraries
import numpy as np
import pandas as pd
import timeit

# Importing the dataset functions
from scripts.utils import mixcr_input
from scripts.utils import data_split
from scripts.utils import data_split_adj
import time

# Load MiXCR output txt files; Assign class labels (0=non-binder, 1=binder)
mHER_H3_1_Ab = mixcr_input('data/mHER_H3_1_Ab.txt', 0)
mHER_H3_1_AgN = mixcr_input('data/mHER_H3_1_AgN.txt', 0)
mHER_H3_1_Ag = mixcr_input('data/mHER_H3_1_1Ag.txt', 1)
mHER_H3_1_Ag647 = mixcr_input('data/mHER_H3_1_2Ag647.txt', 1)
mHER_H3_1_Ag488 = mixcr_input('data/mHER_H3_1_2Ag488.txt', 1)

mHER_H3_2_Ab = mixcr_input('data/mHER_H3_2_Ab.txt', 0)
mHER_H3_2_AgN = mixcr_input('data/mHER_H3_2_AgN.txt', 0)
mHER_H3_2_Ag = mixcr_input('data/mHER_H3_2_1Ag.txt', 1)
mHER_H3_2_Ag647 = mixcr_input('data/mHER_H3_2_2Ag647.txt', 1)
mHER_H3_2_Ag488 = mixcr_input('data/mHER_H3_2_2Ag488.txt', 1)

mHER_H3_3_Ab = mixcr_input('data/mHER_H3_3_Ab.txt', 0)
mHER_H3_3_AgN = mixcr_input('data/mHER_H3_3_AgN.txt', 0)
mHER_H3_3_Ag = mixcr_input('data/mHER_H3_3_1Ag.txt', 1)
mHER_H3_3_Ag647 = mixcr_input('data/mHER_H3_3_2Ag647.txt', 1)
mHER_H3_3_Ag488 = mixcr_input('data/mHER_H3_3_2Ag488.txt', 1)

# Combine the non-binding sequence data sets
# Non-binding data sets include Ab+ data and Ag- sorted data for all 3 libraries
mHER_H3_AgNeg = pd.concat([mHER_H3_1_Ab, mHER_H3_1_AgN, mHER_H3_2_Ab, mHER_H3_2_AgN, mHER_H3_3_Ab, mHER_H3_3_AgN])
mHER_H3_AgNeg = mHER_H3_AgNeg.drop_duplicates(subset='AASeq')

# Combine the binding sequence data sets
# Binding data sets include Ag+ data from 2 rounds of Ag enrichment for all 3 libraries; Ag+ data from 1 round is omitted
mHER_H3_AgPos = pd.concat([mHER_H3_1_Ag647, mHER_H3_1_Ag488, mHER_H3_2_Ag647, mHER_H3_2_Ag488, mHER_H3_3_Ag647, mHER_H3_3_Ag488])
mHER_H3_AgPos = mHER_H3_AgPos.drop_duplicates(subset='AASeq')

# Combine the binding/non-binding data sets and split into training and test sets
mHER_H3_all = data_split(mHER_H3_AgPos, mHER_H3_AgNeg)
mHER_H3_all.train = mHER_H3_all.train.sample(frac=1).reset_index(drop=True)
mHER_H3_all.test = mHER_H3_all.test.sample(frac=1).reset_index(drop=True)
mHER_H3_all.val = mHER_H3_all.val.sample(frac=1).reset_index(drop=True)

# Import scripts for training/testing LSTM-RNN and CNN classification models
from scripts.ANN import ANN_classification
from scripts.RNN import RNN_classification
from scripts.CNN import CNN_classification
from scripts.SVM import SVM_classification
from scripts.RF import RF_classification
from scripts.KNN import KNN_classification

# Train and test LSTM-RNN and CNN with unadjusted (class split) data set
ANN_all, ANN_times = ANN_classification(mHER_H3_all, 0.31, 'All_data')

"""RNN_all, RNN_times = RNN_classification(mHER_H3_all, 0.31, 'All_data')"""

CNN_all, CNN_times = CNN_classification(mHER_H3_all, 0.31, 'All_data')
"""SVM_all, SVM_time = SVM_classification(mHER_H3_all, 0.31, 'All_data')
RF_all, RF_time = RF_classification(mHER_H3_all, 0.31, 'All_data')
KNN_all, KNN_time = KNN_classification(mHER_H3_all, 0.31, 'All_data')"""


## save model !! 

from keras.models import load_model
"""RNN_all = load_model('models/RNN_HER2.h5')"""
CNN_all = load_model('models/CNN_HER2.h5')

# Import in silico sequence generator and classifier
from scripts.utils import seq_classification

# Define the amino acids per position to be used in the in silico library
AA_per_pos =\
[['Y', 'W'],\
['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],\
['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],\
['A', 'D', 'F', 'G', 'H', 'I', 'L', 'N', 'P', 'R', 'S', 'T', 'V', 'Y'],\
['A', 'G', 'H', 'S',],\
['F', 'L'],\
['Y'],\
['A', 'E', 'K', 'L', 'P', 'Q', 'T', 'V'],\
['F', 'H', 'I', 'L', 'N', 'Y'],\
['A', 'D', 'E', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'T', 'V']]

# Produce all CDRH3 sequences and their prediction values if P(binder) > 0.5
RNN_all_seq, RNN_all_pred = seq_classification(AA_per_pos, RNN_all)
CNN_all_seq, CNN_all_pred = seq_classification(AA_per_pos, CNN_all)

# Write output to .csv file
RNN_all_df = pd.DataFrame({'AASeq': RNN_all_seq, 'Pred': RNN_all_pred}, columns=['AASeq', 'Pred'])
RNN_all_df.to_csv('data/RNN_H3_all_2020-03.csv', sep=',')

CNN_all_df = pd.DataFrame({'AASeq': CNN_all_seq, 'Pred': CNN_all_pred}, columns=['AASeq', 'Pred'])
CNN_all_df.to_csv('data/CNN_H3_all_2020-03.csv', sep=',')








## Testing for model execution
## Load in 100k random sequences
from scripts.utils import seq_class_test
from scripts.utils import seq_class_test2
from scripts.utils import seq_class_test3

rand_100 = pd.read_csv('data/Opt-Negs.csv', header=None)
rand_100 = pd.read_csv('data/sim_repertoire_random.csv', header=None)
rand_100.columns = ['AASeq']
AA_list = rand_100.loc[:, 'AASeq'].values

start_time = time.time()
seq_class_test3(AA_list, ANN_all)
end_time = time.time()
ANN_exe_time = end_time - start_time
 
start_time = time.time()
rnn_a, rnn_b, rnn_c = seq_class_test(AA_list, RNN_all)
end_time = time.time()
RNN_exe_time = end_time - start_time

start_time = time.time()
cnn_a, cnn_b, cnn_c = seq_class_test(AA_list, CNN_all)
end_time = time.time()
CNN_exe_time = end_time - start_time

start_time = time.time()
seq_class_test2(AA_list, SVM_all)
end_time = time.time()
SVM_exe_time = end_time - start_time

start_time = time.time()
seq_class_test2(AA_list, RF_all)
end_time = time.time()
RF_exe_time = end_time - start_time

start_time = time.time()
seq_class_test2(AA_list, KNN_all)
end_time = time.time()
KNN_exe_time = end_time - start_time