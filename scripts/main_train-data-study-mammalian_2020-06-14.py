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

# Shuffle sequences
mHER_H3_AgNeg = mHER_H3_AgNeg.sample(frac=1)
mHER_H3_AgNeg.reset_index(drop=True, inplace=True)

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

# Shuffle sequences
mHER_H3_AgPos = mHER_H3_AgPos.sample(frac=1)
mHER_H3_AgPos.reset_index(drop=True, inplace=True)


# ----------------------
# Load and prepare data
# ----------------------

# Create collection with training and test split
mHER_all, unused_seq = data_split_adj(mHER_H3_AgPos, mHER_H3_AgNeg, 0.5)

# TODO: I think the copy statements can be replaced by pandas.copy

# Create shallow copy of the data collection
mHER_yeast_sub = copy.copy(mHER_all)

# TODO: Clean those columns, what do they do?

ML_columns = ('Train_size', 'LogReg_acc', 'LogReg_prec', 'LogReg_recall',
              'LogReg_train_time', 'LogReg_test_time', 'LogReg2D_acc',
              'LogReg2D_prec', 'LogReg2D_recall', 'LogReg2D_train_time',
              'LogReg2D_test_time', 'KNN_acc', 'KNN_prec', 'KNN_recall',
              'KNN_train_time', 'KNN_test_time', 'LSVM_acc', 'LSVM_prec',
              'LSVM_recall', 'LSVM_train_time', 'LSVM_test_time', 'SVM_acc',
              'SVM_prec', 'SVM_recall', 'SVM_train_time', 'SVM_test_time',
              'RF_acc', 'RF_prec', 'RF_recall', 'RF_train_time',
              'RF_test_time', 'ANN_acc', 'ANN_prec', 'ANN_recall',
              'ANN_train_time', 'ANN_test_time', 'CNN_acc', 'CNN_prec',
              'CNN_recall', 'CNN_train_time', 'CNN_test_time', 'RNN_acc',
              'RNN_prec', 'RNN_recall', 'RNN_train_time', 'RNN_test_time')

ML_models = pd.DataFrame(columns=ML_columns)

# TODO: Fix range here
for x in range(1, 9):
    # Hyperparmameter definition
    y = 2**x
    z = 100*y
    if z > 15000:
        z = 15000

    # Sample z sequences from the training set
    mHER_yeast_sub.train = copy.copy(mHER_all.train)
    mHER_yeast_sub.train = mHER_yeast_sub.train.sample(n=z)
    mHER_yeast_sub.train.reset_index(drop=True, inplace=True)

    # Select test data
    ML_df = mHER_yeast_sub.test

    # TODO: This can be done nicer!
    LogReg_pred, LogReg_stats = LogReg_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['LogReg_pred'] = LogReg_pred

    LogReg2D_pred, LogReg2D_stats = LogReg2D_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['LogReg2D_pred'] = LogReg2D_pred

    KNN_pred, KNN_stats = KNN_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['KNN_pred'] = KNN_pred

    LSVM_pred, LSVM_stats = LSVM_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['LSVM_pred'] = LSVM_pred

    SVM_pred, SVM_stats = SVM_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['SVM_pred'] = SVM_pred

    RF_pred, RF_stats = RF_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['RF_pred'] = RF_pred

    ANN_pred, ANN_stats = ANN_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['ANN_pred'] = ANN_pred

    CNN_pred, CNN_stats = CNN_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['CNN_pred'] = CNN_pred

    RNN_pred, RNN_stats = RNN_classification(
        mHER_yeast_sub, z, '{}'.format(z)
    )
    ML_df['RNN_pred'] = RNN_pred

    # TODO: Make the directory before saving it!!
    ML_df.to_csv(
        'figures/ML_pred_data_TrainSize_14Jun_{}.csv'.format(z)
    )
    all_stats = np.concatenate(
        (np.array([x]), LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats,
         SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats)
    )
    ML_models = ML_models.append(
        pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True
    )

ML_models.to_csv('figures/ML_model_stats_TrainSize_14Jun.csv')

# TODO: Imports at the beginning

mHER_all, unused_seq = data_split_adj(mHER_H3_AgPos, mHER_H3_AgNeg, 0.5)


# TODO: Why is this copied again?
mHER_all_copy = copy.copy(mHER_all)

# TODO: I think this is duplicatd as well?

ML_columns = ('Train_size', 'LogReg_acc', 'LogReg_prec', 'LogReg_recall', 'LogReg_train_time', 'LogReg_test_time', 'LogReg2D_acc', 'LogReg2D_prec', 'LogReg2D_recall', 'LogReg2D_train_time', 'LogReg2D_test_time', 'KNN_acc', 'KNN_prec', 'KNN_recall', 'KNN_train_time', 'KNN_test_time', 'LSVM_acc', 'LSVM_prec', 'LSVM_recall', 'LSVM_train_time', 'LSVM_test_time',
              'SVM_acc', 'SVM_prec', 'SVM_recall', 'SVM_train_time', 'SVM_test_time', 'RF_acc', 'RF_prec', 'RF_recall', 'RF_train_time', 'RF_test_time', 'ANN_acc', 'ANN_prec', 'ANN_recall', 'ANN_train_time', 'ANN_test_time', 'CNN_acc', 'CNN_prec', 'CNN_recall', 'CNN_train_time', 'CNN_test_time', 'RNN_acc', 'RNN_prec', 'RNN_recall', 'RNN_train_time', 'RNN_test_time')
ML_models = pd.DataFrame(columns=('Train_size', 'LogReg_acc', 'LogReg_prec', 'LogReg_recall', 'LogReg_train_time', 'LogReg_test_time', 'LogReg2D_acc', 'LogReg2D_prec', 'LogReg2D_recall', 'LogReg2D_train_time', 'LogReg2D_test_time', 'KNN_acc', 'KNN_prec', 'KNN_recall', 'KNN_train_time', 'KNN_test_time', 'LSVM_acc', 'LSVM_prec', 'LSVM_recall', 'LSVM_train_time', 'LSVM_test_time',
                                  'SVM_acc', 'SVM_prec', 'SVM_recall', 'SVM_train_time', 'SVM_test_time', 'RF_acc', 'RF_prec', 'RF_recall', 'RF_train_time', 'RF_test_time', 'ANN_acc', 'ANN_prec', 'ANN_recall', 'ANN_train_time', 'ANN_test_time', 'CNN_acc', 'CNN_prec', 'CNN_recall', 'CNN_train_time', 'CNN_test_time', 'RNN_acc', 'RNN_prec', 'RNN_recall', 'RNN_train_time', 'RNN_test_time'))

# TODO: Fix the range!
for x in np.linspace(0, 10000, 11):
    x = int(x)
    print(x)  # TODO: Remove print statement!

    # TODO: Fix this with pandas!
    mHER_all_copy.train = pd.concat(
        [copy.copy(mHER_all.train), unused_seq[0:x]])
    mHER_all_copy.train = mHER_all_copy.train.sample(
        frac=1).reset_index(drop=True)
    mHER_all_copy.test = copy.copy(mHER_all.test)
    mHER_all_copy.val = copy.copy(mHER_all.val)

    ML_df = mHER_all_copy.test

    LogReg_pred, LogReg_stats = LogReg_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['LogReg_pred'] = LogReg_pred

    LogReg2D_pred, LogReg2D_stats = LogReg2D_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['LogReg2D_pred'] = LogReg2D_pred

    KNN_pred, KNN_stats = KNN_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['KNN_pred'] = KNN_pred

    LSVM_pred, LSVM_stats = LSVM_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['LSVM_pred'] = LSVM_pred

    SVM_pred, SVM_stats = SVM_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['SVM_pred'] = SVM_pred

    RF_pred, RF_stats = RF_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['RF_pred'] = RF_pred

    ANN_pred, ANN_stats = ANN_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['ANN_pred'] = ANN_pred

    CNN_pred, CNN_stats = CNN_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['CNN_pred'] = CNN_pred

    RNN_pred, RNN_stats = RNN_classification(
        mHER_all_copy, x, '{}'.format(x)
    )
    ML_df['RNN_pred'] = RNN_pred

    # TODO: Lots of code here is duplicated! Functionalize it!!
    ML_df.to_csv('figures/ML_increase_negs_{}.csv'.format(x))
    all_stats = np.concatenate(
        (np.array([x]), LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats,
         SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats)
    )
    ML_models = ML_models.append(
        pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True
    )

ML_models.to_csv('figures/ML_incrase_negs_combined.csv')
