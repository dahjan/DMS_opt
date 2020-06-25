#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:58 2019

@author: dmason
"""

# Import libraries
import numpy as np
import pandas as pd
import copy

# Import machine learning models
from LogReg5 import LogReg_classification
from LogReg2D5 import LogReg2D_classification
from KNN5 import KNN_classification
from LSVM5 import LSVM_classification
from SVM5 import SVM_classification
from RF5 import RF_classification
from ANN5 import ANN_classification
from CNN5 import CNN_classification
from RNN5 import RNN_classification

# Import custom functions
from utils import mixcr_input, data_split_adj


# ----------------------
# Load MiXCR output
# ----------------------

# Class labels:
# antigen binder = 1, non-binder = 0

# TODO: Functionalize this!!

# Combine the non-binding sequence data sets.
# Non-binding data sets include Ab+ data and Ag-
# sorted data for all 3 libraries
ab_neg_files = [
    'mHER_H3_1_Ab.txt', 'mHER_H3_1_AgN.txt',
    'mHER_H3_2_Ab.txt', 'mHER_H3_2_AgN.txt',
    'mHER_H3_3_Ab.txt', 'mHER_H3_3_AgN.txt'
]
ab_neg_data = []
for file in ab_neg_files:
    ab_neg_data.append(
        mixcr_input('data/' + file, Ag_class=0)
    )
mHER_H3_AgNeg = pd.concat(ab_neg_data)

# Drop duplicate sequences
mHER_H3_AgNeg = mHER_H3_AgNeg.drop_duplicates(subset='AASeq')

# Remove 'CAR/CSR' motif and last amino acid
mHER_H3_AgNeg['AASeq'] = [x[3:-1] for x in mHER_H3_AgNeg['AASeq']]

# Shuffle sequences and reset index
mHER_H3_AgNeg = mHER_H3_AgNeg.sample(frac=1).reset_index(drop=True)

# TODO: Functionalize this!!

# Combine the binding sequence data sets
# Binding data sets include Ag+ data from 2 rounds of Ag enrichment for all
# 3 libraries; Ag+ data from 1 round is omitted
ab_pos_files = [
    'mHER_H3_1_2Ag647.txt', 'mHER_H3_1_2Ag488.txt',
    'mHER_H3_2_2Ag647.txt', 'mHER_H3_2_2Ag488.txt',
    'mHER_H3_3_2Ag647.txt', 'mHER_H3_3_2Ag488.txt'
]
ab_pos_data = []
for file in ab_pos_files:
    ab_pos_data.append(
        mixcr_input('data/' + file, Ag_class=1)
    )
mHER_H3_AgPos = pd.concat(ab_pos_data)

# Drop duplicate sequences
mHER_H3_AgPos = mHER_H3_AgPos.drop_duplicates(subset='AASeq')

# Remove 'CAR/CSR' motif and last amino acid
mHER_H3_AgPos['AASeq'] = [x[3:-1] for x in mHER_H3_AgPos['AASeq']]

# Shuffle sequences and reset index
mHER_H3_AgPos = mHER_H3_AgPos.sample(frac=1).reset_index(drop=True)

# Create collection with training and test split
mHER_all, unused_seq = data_split_adj(mHER_H3_AgPos, mHER_H3_AgNeg, ratio=0.5)

# Create shallow copy of the data collection
mHER_all_copy = copy.copy(mHER_all)


# ----------------------
# Run classifiers
# ----------------------

ML_columns = ('Train_size', 'LogReg_pred', 'LogReg2D_pred',
              'KNN_pred', 'LSVM_pred', 'SVM_pred', 'RF_pred',
              'ANN_pred', 'CNN_pred', 'RNN_pred')

ML_df = pd.DataFrame(columns=ML_columns)

# Add unused sequences to training set
for x in np.linspace(0, 10000, 11):
    x = int(x)

    # Add x unused sequences to training set
    mHER_all_copy.train = pd.concat(
        [copy.copy(mHER_all.train), unused_seq[0:x]]
    )

    # Shuffle training data
    mHER_all_copy.train = mHER_all_copy.train.sample(
        frac=1
    ).reset_index(drop=True)
    mHER_all_copy.test = copy.copy(mHER_all.test)
    mHER_all_copy.val = copy.copy(mHER_all.val)

    # Run all classifiers
    LogReg_pred = LogReg_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LogReg2D_pred = LogReg2D_classification(
        mHER_all_copy, '{}'.format(x)
    )
    KNN_pred = KNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    LSVM_pred = LSVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    SVM_pred = SVM_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RF_pred = RF_classification(
        mHER_all_copy, '{}'.format(x)
    )
    ANN_pred = ANN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    CNN_pred = CNN_classification(
        mHER_all_copy, '{}'.format(x)
    )
    RNN_pred = RNN_classification(
        mHER_all_copy, '{}'.format(x)
    )

    # Append a row with all predictions
    all_preds = np.concatenate(
        (np.array([x]), LogReg_pred, LogReg2D_pred, KNN_pred, LSVM_pred,
         SVM_pred, RF_pred, ANN_pred, CNN_pred, RNN_pred)
    )
    ML_df = ML_df.append(
        pd.DataFrame([all_preds], columns=list(ML_columns)), ignore_index=True
    )

ML_df.to_csv('figures/ML_increase_negs_all_preds.csv')
