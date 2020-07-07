#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:58 2019

@author: dmason
"""

# Import libraries
import os
import copy
import numpy as np
import pandas as pd

# Import machine learning models
from LogReg import LogReg_classification
from LogReg2D import LogReg2D_classification
from KNN import KNN_classification
from LSVM import LSVM_classification
from SVM import SVM_classification
from RF import RF_classification
from ANN import ANN_classification
from CNN import CNN_classification
from RNN import RNN_classification

# Import custom functions
from utils import data_split, data_split_adj, \
    seq_classification, load_input_data


# ----------------------
# Load input data
# ----------------------

# Class labels:
# antigen binder = 1, non-binder = 0

# Load non-binding sequences
ab_neg_files = [
    'mHER_H3_1_Ab.txt', 'mHER_H3_1_AgN.txt',
    'mHER_H3_2_Ab.txt', 'mHER_H3_2_AgN.txt',
    'mHER_H3_3_Ab.txt', 'mHER_H3_3_AgN.txt'
]
mHER_H3_AgNeg = load_input_data(ab_neg_files, Ag_class=0)

# Load binding sequences
ab_pos_files = [
    'mHER_H3_1_2Ag647.txt', 'mHER_H3_1_2Ag488.txt',
    'mHER_H3_2_2Ag647.txt', 'mHER_H3_2_2Ag488.txt',
    'mHER_H3_3_2Ag647.txt', 'mHER_H3_3_2Ag488.txt'
]
mHER_H3_AgPos = load_input_data(ab_pos_files, Ag_class=1)

# Save those files
mHER_H3_AgNeg.to_csv('data/mHER_H3_AgNeg.csv')
mHER_H3_AgPos.to_csv('data/mHER_H3_AgPos.csv')


# ----------------------
# Run classifiers
# ----------------------

# Create collection with training and test split
mHER_all_adj, unused_seq = data_split_adj(
    mHER_H3_AgPos, mHER_H3_AgNeg, fraction=0.5
)

# Create shallow copy of the data collection
mHER_all_copy = copy.copy(mHER_all_adj)

# Create directory to store figures (hard-coded!)
os.makedirs('figures', exist_ok=True)

# Create columns for final dataframe
ML_columns = ('Train_size', 'LogReg_acc', 'LogReg_prec', 'LogReg_recall',
              'LogReg_F1', 'LogReg_MCC', 'LogReg2D_acc', 'LogReg2D_prec',
              'LogReg2D_recall', 'LogReg2D_F1', 'LogReg2D_MCC',
              'KNN_acc', 'KNN_prec', 'KNN_recall', 'KNN_F1', 'KNN_MCC',
              'LSVM_acc', 'LSVM_prec', 'LSVM_recall', 'LSVM_F1', 'LSVM_MCC',
              'SVM_acc', 'SVM_prec', 'SVM_recall', 'SVM_F1', 'SVM_MCC',
              'RF_acc', 'RF_prec', 'RF_recall', 'RF_F1', 'RF_MCC',
              'ANN_acc', 'ANN_prec', 'ANN_recall', 'ANN_F1', 'ANN_MCC',
              'CNN_acc', 'CNN_prec', 'CNN_recall', 'CNN_F1', 'CNN_MCC',
              'RNN_acc', 'RNN_prec', 'RNN_recall', 'RNN_F1', 'RNN_MCC')
ML_df = pd.DataFrame(columns=ML_columns)

# Add unused sequences to training set
for x in np.linspace(0, 10000, 11):
    x = int(x)

    # Add x unused sequences to training set
    mHER_all_copy.train = pd.concat(
        [copy.copy(mHER_all_adj.train), unused_seq[0:x]]
    )

    # Shuffle training data
    mHER_all_copy.train = mHER_all_copy.train.sample(
        frac=1
    ).reset_index(drop=True)
    mHER_all_copy.test = copy.copy(mHER_all_adj.test)
    mHER_all_copy.val = copy.copy(mHER_all_adj.val)

    # Run all classifiers
    LogReg_stats = LogReg_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LogReg2D_stats = LogReg2D_classification(
        mHER_all_copy, '{}'.format(x)
    )
    KNN_stats = KNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LSVM_stats = LSVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    SVM_stats = SVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RF_stats = RF_classification(
        mHER_all_copy, '{}'.format(x)
    )
    ANN_stats = ANN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    CNN_stats = CNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RNN_stats = RNN_classification(
        mHER_all_copy, '{}'.format(x)
    )

    # Append a row with all statistics
    all_stats = np.concatenate(
        (np.array([x]), LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats,
         SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats)
    )
    ML_df = ML_df.append(
        pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True
    )

# Save statistics to file
ML_df.to_csv('figures/ML_increase_negs_combined.csv')


# ----------------------
# Run classifiers on in
# silico generated data
# ----------------------

# Create collection with training and test split
mHER_H3_all = data_split(mHER_H3_AgPos, mHER_H3_AgNeg)

# Create model directory
model_dir = 'classification'
os.makedirs(model_dir, exist_ok=True)

# Use tuned model parameters for CNN (performed in separate script)
params = [['CONV', 400, 5, 1],
          ['DROP', 0.2],
          ['POOL', 2, 1],
          ['FLAT'],
          ['DENSE', 300]]

# Train and test ANN and CNN with unadjusted (class split) data set
ANN_all = ANN_classification(
    mHER_H3_all, 'All_data', save_model=model_dir
)
CNN_all = CNN_classification(
    mHER_H3_all, 'All_data', save_model=model_dir, params=params
)
# RNN_all = RNN_classification(
#     mHER_H3_all, 'All_data', save_model=model_dir
# )

# Generate CDRH3 sequences in silico and calculate their
# prediction values if P(binder) > 0.5
print('[INFO] Classifying in silico generated sequences')
ANN_all_seq, ANN_all_pred = seq_classification(ANN_all, flatten_input=True)
CNN_all_seq, CNN_all_pred = seq_classification(CNN_all)
# RNN_all_seq, RNN_all_pred = seq_classification(RNN_all)
print('[INFO] Done')

# Write output to .csv file
ANN_all_df = pd.DataFrame(
    {'AASeq': ANN_all_seq, 'Pred': ANN_all_pred}, columns=['AASeq', 'Pred']
)
ANN_all_df.to_csv(
    os.path.join(model_dir, 'ANN_H3_all.csv')
)

CNN_all_df = pd.DataFrame(
    {'AASeq': CNN_all_seq, 'Pred': CNN_all_pred}, columns=['AASeq', 'Pred']
)
CNN_all_df.to_csv(
    os.path.join(model_dir, 'CNN_H3_all.csv')
)
