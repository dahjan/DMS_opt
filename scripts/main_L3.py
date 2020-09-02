#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 31 16:55:32 2020

@author: djan
"""

# Import libraries
import os
import copy

# Import machine learning models
from CNN import CNN_classification_L3

# Import custom functions
from utils import data_split_L3, load_input_data_L3


# ----------------------
# Load input data
# ----------------------

# Class labels:
# antigen binder = 1, non-binder = 0

# Load non-binding sequences
ab_neg_files = [
    'mHER_L3_3_Ab.txt', 'mHER_L3_3_AgN.txt'
]
mHER_L3_AgNeg = load_input_data_L3(ab_neg_files, Ag_class=0)

# Load binding sequences
ab_pos_files = [
    'mHER_L3_3_2Ag647.txt', 'mHER_L3_3_2Ag488.txt'
]
mHER_L3_AgPos = load_input_data_L3(ab_pos_files, Ag_class=1)

# Save those files
mHER_L3_AgNeg.to_csv('data/mHER_L3_AgNeg.csv')
mHER_L3_AgPos.to_csv('data/mHER_L3_AgPos.csv')


# ----------------------
# Run classifiers on
# CDR-L3 sequence data
# ----------------------

# Create directory to store figures (hard-coded!)
os.makedirs('figures', exist_ok=True)

# Create collection with training and test split
mHER_L3_all = data_split_L3(mHER_L3_AgPos, mHER_L3_AgNeg)

# Create model directory
model_dir = 'classification'
os.makedirs(model_dir, exist_ok=True)

# Use tuned model parameters for CNN (performed in separate script)
params = [['CONV', 64, 2, 1],
          ['DROP', 0.5],
          ['POOL', 2, 1],
          ['FLAT'],
          ['DENSE', 32]]

# Train and test CNN with unadjusted (class split) data set
_ = CNN_classification_L3(
    mHER_L3_all, 'All_data', save_model=model_dir, params=params
)


# ----------------------
# Run classifiers on
# randomly shuffled data
# ----------------------

# Create a deep copy of the data collection
mHER_all_copy = copy.deepcopy(mHER_L3_all)

# Randomly shuffle labels
mHER_all_copy.train["AgClass"] = mHER_L3_all.train.AgClass.sample(
    frac=1).values
mHER_all_copy.test["AgClass"] = mHER_L3_all.test.AgClass.sample(frac=1).values

# Train and test CNN with randomly shuffled data
_ = CNN_classification_L3(
    mHER_all_copy, 'Shuffled_data', save_model=model_dir, params=params
)
