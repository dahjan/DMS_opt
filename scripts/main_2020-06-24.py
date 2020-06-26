#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:58 2019

@author: dmason
"""

# Importing libraries
import os
import pandas as pd

# Import machine learning models
from ANN5 import ANN_classification
from CNN5 import CNN_classification

# Import custom functions
from utils import mixcr_input, data_split, seq_classification


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

# Remove 'CAR/CSR' motif and last two amino acids
mHER_H3_AgNeg['AASeq'] = [x[3:-2] for x in mHER_H3_AgNeg['AASeq']]

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

# Remove 'CAR/CSR' motif and last two amino acids
mHER_H3_AgPos['AASeq'] = [x[3:-2] for x in mHER_H3_AgPos['AASeq']]

# Shuffle sequences and reset index
mHER_H3_AgPos = mHER_H3_AgPos.sample(frac=1).reset_index(drop=True)

# Create collection with training and test split
mHER_H3_all = data_split(mHER_H3_AgPos, mHER_H3_AgNeg)

# ----------------------
# Run classifiers on in
# silico generated data
# ----------------------

# Create model directory
model_dir = 'model_out'
os.makedirs(model_dir, exist_ok=True)

# Train and test ANN and CNN with unadjusted (class split) data set
ANN_all = ANN_classification(
    mHER_H3_all, 'All_data', save_model=model_dir
)
CNN_all = CNN_classification(
    mHER_H3_all, 'All_data', save_model=model_dir
)

# Generate CDRH3 sequences in silico and calculate their
# prediction values if P(binder) > 0.5
ANN_all_seq, ANN_all_pred = seq_classification(ANN_all)
CNN_all_seq, CNN_all_pred = seq_classification(CNN_all)

# Write output to .csv file
ANN_all_df = pd.DataFrame(
    {'AASeq': ANN_all_seq, 'Pred': ANN_all_pred}, columns=['AASeq', 'Pred']
)
ANN_all_df.to_csv('data/CNN_H3_all_2020-03.csv', sep=',')

CNN_all_df = pd.DataFrame(
    {'AASeq': CNN_all_seq, 'Pred': CNN_all_pred}, columns=['AASeq', 'Pred']
)
CNN_all_df.to_csv('data/CNN_H3_all_2020-03.csv', sep=',')
