#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 28 10:54:04 2020

@author: Jan Dahinden
"""

# Import libraries
import os
import keras
import numpy as np
import pandas as pd
from Bio.Alphabet import IUPAC
from sklearn.model_selection import RandomizedSearchCV

# Import custom functions
from utils import one_hot_encoder, load_input_data, \
    build_classifier


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

# Fuse Ag positive and negative sequences
Ag_combined = pd.concat([mHER_H3_AgPos, mHER_H3_AgNeg])
Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

# Save sequences and labels
X = Ag_combined.AASeq
label = Ag_combined.AgClass

# One hot encode the sequences
X_ohe = [one_hot_encoder(s=seq, alphabet=IUPAC.protein) for seq in X]
X_ohe = np.transpose(np.asarray(X_ohe), (0, 2, 1))


# ----------------------
# Tuning CNN classifier
# ----------------------

# Create output directory
model_dir = 'model_tuning'
os.makedirs(model_dir, exist_ok=True)

# Build the classifier
classifier = keras.wrappers.scikit_learn.KerasClassifier(
    build_fn=build_classifier
)

# Parameters to tune
parameters = {'batch_size': [16, 32],
              'nb_epoch': [10, 20, 30],
              'filters': [200, 400, 800],
              'kernels': [3, 5],
              'strides': [1, 2],
              'activation': ['sigmoid', 'relu'],
              'dropout': [0.2, 0.5],
              'dense': [50, 100, 200],
              'optimizer': ['adam', 'SGD']}

# Run randomized search on all CPUs
rand_search = RandomizedSearchCV(estimator=classifier,
                                 param_distributions=parameters,
                                 n_jobs=-1,
                                 return_train_score=True)
rand_search.fit(X_ohe, label)

# Save best performing model
with open(os.path.join(model_dir, 'tuned_CNN_model.txt'), 'w') as f:
    f.write('Best score: %.2f \n\n' % rand_search.best_score_)
    f.write('Best parameters: \n')
    f.write(str(rand_search.best_params_))
