#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:58 2019

@author: dmason
"""

# Importing libraries
import numpy as np
import pandas as pd

# Importing the dataset functions
from scripts.utils import mixcr_input
from scripts.utils2 import data_split
from scripts.utils2 import align_input

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
mHER_H3_AgNeg['AASeq']  = [x[3:-1] for x in mHER_H3_AgNeg['AASeq']]
mHER_H3_AgNeg = mHER_H3_AgNeg.sample(frac=1).reset_index(drop=True)

# Combine the binding sequence data sets
# Binding data sets include Ag+ data from 2 rounds of Ag enrichment for all 3 libraries; Ag+ data from 1 round is omitted
mHER_H3_AgPos = pd.concat([mHER_H3_1_Ag647, mHER_H3_1_Ag488, mHER_H3_2_Ag647, mHER_H3_2_Ag488, mHER_H3_3_Ag647, mHER_H3_3_Ag488])
mHER_H3_AgPos = mHER_H3_AgPos.drop_duplicates(subset='AASeq')
mHER_H3_AgPos['AASeq']  = [x[3:-1] for x in mHER_H3_AgPos['AASeq']]
mHER_H3_AgPos = mHER_H3_AgPos.sample(frac=1).reset_index(drop=True)

from scripts.LogReg5 import LogReg_classification
from scripts.LogReg2D5 import LogReg2D_classification
from scripts.KNN5 import KNN_classification
from scripts.LSVM5 import LSVM_classification
from scripts.SVM5 import SVM_classification
from scripts.RF5 import RF_classification
from scripts.ANN5 import ANN_classification
from scripts.CNN5 import CNN_classification
from scripts.RNN5 import RNN_classification

from scripts.utils3 import data_split_adj
mHER_all, unused_seq = data_split_adj(mHER_H3_AgPos, mHER_H3_AgNeg, 0.5)

import copy
mHER_yeast_sub = copy.copy(mHER_all)

ML_columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time')
ML_models = pd.DataFrame(columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time'))

for x in range(1,9):
    y = 2**x
    z = 100*y
    if z > 15000:
        z = 15000

    mHER_yeast_sub.train = copy.copy(mHER_all.train)
    mHER_yeast_sub.train = mHER_yeast_sub.train.sample(n=z).reset_index(drop=True)
    
    ML_df = mHER_yeast_sub.test
    
    LogReg_pred, LogReg_stats = LogReg_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['LogReg_pred'] = LogReg_pred
    
    LogReg2D_pred, LogReg2D_stats = LogReg2D_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['LogReg2D_pred'] = LogReg2D_pred
    
    KNN_pred, KNN_stats = KNN_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['KNN_pred'] = KNN_pred
    
    LSVM_pred, LSVM_stats = LSVM_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['LSVM_pred'] = LSVM_pred
    
    SVM_pred, SVM_stats = SVM_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['SVM_pred'] = SVM_pred
    
    RF_pred, RF_stats = RF_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['RF_pred'] = RF_pred

    ANN_pred, ANN_stats = ANN_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['ANN_pred'] = ANN_pred

    CNN_pred, CNN_stats = CNN_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['CNN_pred'] = CNN_pred
    
    RNN_pred, RNN_stats = RNN_classification(mHER_yeast_sub, z, '{}'.format(z))
    ML_df['RNN_pred'] = RNN_pred
    
    ML_df.to_csv('figures/ML_pred_data_TrainSize_14Jun_{}.csv'.format(z), sep=',')
    all_stats = np.concatenate((np.array([x]),LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats, SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats))
    ML_models = ML_models.append(pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True)
    
ML_models.to_csv('figures/ML_model_stats_TrainSize_14Jun.csv', sep=',')










"""

import copy
mHER_yeast_sub = copy.copy(mHER_yeast_all)
mHER_yeast_test = mHER_yeast_sub.test.sample(n=30000).reset_index(drop=True)
mHER_yeast_val = mHER_yeast_sub.val.sample(n=30000).reset_index(drop=True)


ML_columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time')
ML_models = pd.DataFrame(columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time'))

for x in np.linspace(0.1,0.9,9):
    
    mHER_yeast = data_split_adj(mHER_yeast_AgH, mHER_yeast_AgNeg, x)
    
    mHER_yeast.train = mHER_yeast.train.sample(n=2000).reset_index(drop=True)
    mHER_yeast.test = copy.copy(mHER_yeast_test)
    mHER_yeast.val = copy.copy(mHER_yeast_val)
    
    ML_df = mHER_yeast.test
    
    KNN_pred, KNN_stats = KNN_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['KNN_pred'] = KNN_pred
    
    LSVM_pred, LSVM_stats = LSVM_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['LSVM_pred'] = LSVM_pred
    
    SVM_pred, SVM_stats = SVM_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['SVM_pred'] = SVM_pred
    
    RF_pred, RF_stats = RF_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['RF_pred'] = RF_pred

    ANN_pred, ANN_stats = ANN_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['ANN_pred'] = ANN_pred

    CNN_pred, CNN_stats = CNN_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['CNN_pred'] = CNN_pred
    
    RNN_pred, RNN_stats = RNN_classification(mHER_yeast, x, '{}'.format(x))
    ML_df['RNN_pred'] = RNN_pred
    
    ML_df.to_csv('figures/ML_pred_data_TrainSplit_High_2k_{}.csv'.format(x), sep=',')
    all_stats = np.concatenate((np.array([x]), KNN_stats, LSVM_stats, SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats))
    ML_models = ML_models.append(pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True)
    
ML_models.to_csv('figures/ML_model_stats_TrainSplit_High_2k.csv', sep=',')

"""



from scripts.utils3 import data_split_adj
mHER_all, unused_seq = data_split_adj(mHER_H3_AgPos, mHER_H3_AgNeg, 0.5)

import copy
mHER_all_copy = copy.copy(mHER_all)



ML_columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time')
ML_models = pd.DataFrame(columns = ('Train_size','LogReg_acc','LogReg_prec','LogReg_recall','LogReg_train_time','LogReg_test_time','LogReg2D_acc','LogReg2D_prec','LogReg2D_recall','LogReg2D_train_time','LogReg2D_test_time','KNN_acc','KNN_prec','KNN_recall','KNN_train_time','KNN_test_time','LSVM_acc','LSVM_prec','LSVM_recall','LSVM_train_time','LSVM_test_time','SVM_acc','SVM_prec','SVM_recall','SVM_train_time','SVM_test_time','RF_acc','RF_prec','RF_recall','RF_train_time','RF_test_time','ANN_acc','ANN_prec','ANN_recall','ANN_train_time','ANN_test_time','CNN_acc','CNN_prec','CNN_recall','CNN_train_time','CNN_test_time','RNN_acc','RNN_prec','RNN_recall','RNN_train_time','RNN_test_time'))

for x in np.linspace(0,10000,11):
    x = int(x)
    print(x)
    
    mHER_all_copy.train = pd.concat([copy.copy(mHER_all.train), unused_seq[0:x]])
    mHER_all_copy.train = mHER_all_copy.train.sample(frac=1).reset_index(drop=True)
    mHER_all_copy.test = copy.copy(mHER_all.test)
    mHER_all_copy.val = copy.copy(mHER_all.val)
    
    ML_df = mHER_all_copy.test
    
    LogReg_pred, LogReg_stats = LogReg_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['LogReg_pred'] = LogReg_pred
    
    LogReg2D_pred, LogReg2D_stats = LogReg2D_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['LogReg2D_pred'] = LogReg2D_pred
    
    KNN_pred, KNN_stats = KNN_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['KNN_pred'] = KNN_pred
    
    LSVM_pred, LSVM_stats = LSVM_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['LSVM_pred'] = LSVM_pred
    
    SVM_pred, SVM_stats = SVM_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['SVM_pred'] = SVM_pred
    
    RF_pred, RF_stats = RF_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['RF_pred'] = RF_pred

    ANN_pred, ANN_stats = ANN_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['ANN_pred'] = ANN_pred

    CNN_pred, CNN_stats = CNN_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['CNN_pred'] = CNN_pred
    
    RNN_pred, RNN_stats = RNN_classification(mHER_all_copy, x, '{}'.format(x))
    ML_df['RNN_pred'] = RNN_pred
    
    ML_df.to_csv('figures/ML_increase_negs_{}.csv'.format(x), sep=',')
    all_stats = np.concatenate((np.array([x]),LogReg_stats, LogReg2D_stats, KNN_stats, LSVM_stats, SVM_stats, RF_stats, ANN_stats, CNN_stats, RNN_stats))
    ML_models = ML_models.append(pd.DataFrame([all_stats], columns=list(ML_columns)), ignore_index=True)
    
ML_models.to_csv('figures/ML_incrase_negs_combined.csv', sep=',')