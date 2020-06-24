#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 08:52:23 2019

@author: dmason, frsimon
"""

import pandas as pd
import numpy as np
from Bio import motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, average_precision_score
from inspect import signature
import matplotlib.pyplot as plt



def create_cnn(units_per_layer, activation, regularizer):
    """
    Generate the CNN layers with Keras wrapper
    
    Parameters
    ---
    units_per_layer: architecture features in list format, i.e.:
        Filter information: [CONV, # filters, kernel size, stride]
        Max Pool information: [POOL, pool size, stride]
        Dropout information: [DROP, dropout rate]
        Flatten: [FLAT]
        Dense layer: [DENSE, number nodes]
    
    Activation: Activation function, i.e. ReLU, softmax
    
    Regularizer: Kernel and bias regularizer in convulational and dense
        layers, i.e., regularizers.l1(0.01)
        
    """
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer((10,20)))
    
    # Build network
    
    print('Building CNN network with the following architecture: {}\n'.format(
        units_per_layer))
    
    for i, units in enumerate(units_per_layer):
        
        if units[0] == 'CONV':
            model.add(keras.layers.Conv1D(filters=units[1],
                                          kernel_size=units[2],
                                          strides=units[3],
                                          activation=activation,
                                          kernel_regularizer=regularizer,
                                          bias_regularizer=regularizer,
                                          padding='same'))
        elif units[0] == 'POOL':
            model.add(keras.layers.MaxPool1D(pool_size=units[1], 
                                             strides=units[2]))
        elif units[0] == 'DENSE':
            model.add(keras.layers.Dense(units=units[1],
                                         activation=activation,
                                         kernel_regularizer=regularizer,
                                         bias_regularizer=regularizer))
        elif units[0] == 'DROP':
            model.add(keras.layers.Dropout(rate=units[1]))
        elif units[0] == 'FLAT':
            model.add(keras.layers.Flatten())
        else:
            print('Layer type not implemented')
    
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    
    return model



def mixcr_input(file_name, Ag_class):
    """
    Read in data from the MiXCR TXT output file
    
    Parameters
    ---
    file_name: file name of the MiXCR txt file to read
    
    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1 or non-binder = 0)
               
    """
    
    x = pd.read_table(file_name)
    x = x[["Clone count","Clone fraction","N. Seq. CDR3 ","AA. Seq.CDR3 "]]
    x = x.rename(index=str, columns={"Clone count": "Count", "Clone fraction": "Fraction", "N. Seq. CDR3 ": "NucSeq", "AA. Seq.CDR3 ": "AASeq"})
    x = x[(x.AASeq.str.len()==15) & (x.Count>1)]
    x = x.drop_duplicates(subset='AASeq')
    
    idx = [i for i, aa in enumerate(x['AASeq']) if not '*' in aa]
    x = x.iloc[idx, :]
    idx = [i for i, aa in enumerate(x['AASeq']) if not '_' in aa]
    x = x.iloc[idx, :]
    
    if Ag_class == 0: x['AgClass'] = 0
    if Ag_class == 1: x['AgClass'] = 1
    
    return x



def align_input(file_name, Ag_class):
    """
    Read in data from the MiXCR TXT output file
    
    Parameters
    ---
    file_name: file name of the MiXCR txt file to read
    
    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1 or non-binder = 0)
               
    """
    
    x = pd.read_table(file_name, header=None)
    x.columns = ['AASeq']
    x = x.drop_duplicates(subset='AASeq')
    
    idx = [i for i, aa in enumerate(x['AASeq']) if not '*' in aa]
    x = x.iloc[idx, :]
    idx = [i for i, aa in enumerate(x['AASeq']) if not '_' in aa]
    x = x.iloc[idx, :]
    
    if Ag_class == 0: x['AgClass'] = 0
    if Ag_class == 1: x['AgClass'] = 1
    
    return x


def data_split(Ag_pos, Ag_neg, train_split):
    """
    Create a collection of the data set and split into the training set
    and two test sets. One test set contains the same class split as the
    overall data set, one test set contains a class split of approx.
    10% binders and 90% non-binders.
    
    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    
    """
    
    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
            
    Ag_combined = pd.concat([Ag_pos, Ag_neg])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)
    
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(idx, stratify=Ag_combined['AgClass'], test_size=train_split)

    idx2 = np.arange(0, idx_test.shape[0])

    idx_val, idx_test2 = train_test_split(idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5)
    
    Seq_Ag_data = Collection(train=Ag_combined.iloc[idx_train, :],
                        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
                        test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],
                        complete=Ag_combined)
    
    return Seq_Ag_data



def data_split_adj(Ag_pos, Ag_neg, ratio):
    
    """
    Create a collection of the data set and split into the training set
    and two test sets. Data set is adjusted to the desired class split ratio.
    One test set contains the same class split as the overall data set,
    one test set contains a class split of approx. 10% binders and 90% non-binders.
    
    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    
    """
    
    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    
    # Adjust the length of the data frames to meet the ratio requirements
    
    if len(Ag_pos) <= len(Ag_neg):
        data_size = len(Ag_pos)/ratio
        
        if len(Ag_neg)/(1-ratio) < data_size:
            
            new_data_size = len(Ag_neg)/(1-ratio)
            Ag_pos1 = Ag_pos[0:int((new_data_size*ratio))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((new_data_size*ratio)):len(Ag_pos)]
            
        if len(Ag_neg)/(1-ratio) >= data_size:
        
            Ag_pos1 = Ag_pos[0:int((data_size*(ratio)))]
            Ag_neg1 = Ag_neg[0:int((data_size*(1-ratio)))]
            Unused = pd.concat([Ag_pos[int((data_size*ratio)):len(Ag_pos)], Ag_neg[int((data_size*(1-ratio))):len(Ag_neg)]])
    else:
        data_size = len(Ag_neg)/(1-ratio)
        
        if len(Ag_pos)/(ratio) < data_size:
                            
            new_data_size = len(Ag_pos)/(ratio)
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(new_data_size*(1-ratio)))]
            Unused = Ag_pos[int((new_data_size*ratio)):len(Ag_pos)]
        
        if len(Ag_pos)/(ratio) >= data_size:
        
            Ag_pos1 = Ag_pos[0:int((data_size*(ratio)))]
            Ag_neg1 = Ag_neg[0:int((data_size*(1-ratio)))]
            Unused = pd.concat([Ag_pos[int((data_size*ratio)):len(Ag_pos)], Ag_neg[int((data_size*(1-ratio))):len(Ag_neg)]])
        
    # Combine the positive and negative data frames    
    
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)
    
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(idx, stratify=Ag_combined['AgClass'], test_size=0.3)

    idx2 = np.arange(0, idx_test.shape[0])

    idx_val, idx_test2 = train_test_split(idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5)
    
    """Seq_Ag_data = Collection(train=Ag_combined.iloc[idx_train, :],
                        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
                        test=pd.concat([Ag_combined.iloc[idx_test, :].iloc[idx_test2, :], Unused]),
                        complete=Ag_combined)
    """
    Seq_Ag_data = Collection(train=Ag_combined.iloc[idx_train, :],
                    val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
                    test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],
                    complete=Ag_combined)
        
    return Seq_Ag_data, Unused



def one_hot_encoder(s,  alphabet):
        """
        One hot encodes a biological sequence

        Parameters
        ---
        s: str, sequence which should be encoded
        alphabet: Alphabet object, http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

        Example
        ---
        sequence = 'CARGSSYSSFAYW'
        one_hot_encoder(s=sequence, alphabet=IUPAC.protein)

        Returns
        ---
        x: array, n_size_alphabet, n_length_string
            Sequence as one-hot encoding
        """
        # Build dictionary

        d = {a: i for i, a in enumerate(alphabet.letters)}

        # Encode

        x = np.zeros((len(d), len(s)))
        x[[d[c] for c in s], range(len(s))] = 1

        return x

def one_hot_decoder(x, alphabet):
        """
        Decodes a one-hot encoding to a biological sequence

        Parameters
        ---
        x: array, n_size_alphabet, n_length_string
            Sequence as one-hot encoding
        alphabet: Alphabet object, http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

        Example
        ---
        encoding = one_hot_encoder(sequence, IUPAC.unambiguous_dna)
        one_hot_decoder(encoding, IUPAC.unambiguous_dna)

        Returns
        ---
        s : str, decoded sequence
        """

        d = {a: i for i, a in enumerate(alphabet.letters)}
        inv_d = {i: a for a, i in d.items()}
        s = (''.join(str(inv_d[i]) for i in np.argmax(x, axis=0)))

        return s

def seq_classification(AA_per_pos, classifier):
    
    """
    In silico generate sequences and classify them as a binding or non-binding
    sequence respectively.
    
    Parameters
    ---
    AA_per_pos: amino acids used per position in list format, i.e.,
    [['F','Y','W'],['A','D','G',...'Y']]
    
    classifier: RNN or CNN classification model to use
    
    """

    dim = []
    for x in AA_per_pos:
        dim.append(len(x))
        
        idx = [0]*len(dim)
        pos = 0
        current_seq = np.empty(0, dtype=object)
        pos_seq = np.empty(0, dtype=str)
        pos_pred = np.empty(0, dtype=float)
    
    counter = 1
  
    while(1):
        l = []
        for i in range(0, len(dim)):
            l.append(AA_per_pos[i][idx[i]])
        
        current_seq = np.append(current_seq, (''.join(l)))    
        
                
        if len(current_seq)==500:
            ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in current_seq]
            ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
            seq_pred = classifier.predict(x = ohe_seq)
        
            pos_seq = np.append(pos_seq, current_seq[np.where(seq_pred > 0.50)[0]])
            pos_pred = np.append(pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]])
            # pos_seq.append(current_seq[np.where(seq_pred > 0.5)[0]])
        
            current_seq = np.empty(0, dtype=object)
            print(counter, '/', np.prod(dim)/500)
            counter +=1
            # current_seq = []
       
            
        if sum(idx)==(sum(dim)-len(dim)):
            ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in current_seq]
            ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
            seq_pred = classifier.predict(x = ohe_seq)

            pos_seq = np.append(pos_seq, current_seq[np.where(seq_pred > 0.50)[0]])
            pos_pred = np.append(pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]])
            # pos_seq.append(current_seq[np.where(seq_pred > 0.5)[0]])        
        
            break
        
        
        while(1):
            if (idx[pos]+1) == dim[pos]:
                idx[pos] = 0
                pos +=1
            else:
                idx[pos] +=1
                pos = 0
                break

    return pos_seq, pos_pred
    

def seq_class_test(AA_list, classifier):
    
    aa_test = [x[3:-2] for x in AA_list]
    
    ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in aa_test]
    ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
    seq_pred = classifier.predict(x = ohe_seq)
       
    return seq_pred

def seq_class_test2(AA_list, classifier):
    
    aa_test = [x[3:-2] for x in AA_list]
    
    ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in aa_test]
    ohe_seq = [x.flatten('F') for x in ohe_seq]
    seq_pred = classifier.predict(ohe_seq)       
      
    return seq_pred

def seq_class_test3(AA_list, classifier):
    
    aa_test = [x[3:-2] for x in AA_list]
    
    ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in aa_test]
    ohe_seq = [x.flatten('F') for x in ohe_seq]
    ohe_seq = np.asarray(ohe_seq)
    seq_pred = classifier.predict(ohe_seq)       
      
    return seq_pred


def seq_class_neg(AA_list, classifier):
    
    aa_test = [x[3:-2] for x in AA_list]
    
    ohe_seq = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in aa_test]
    ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
    seq_pred = classifier.predict(x = ohe_seq)
        
    neg_seq = AA_list[np.where(seq_pred <= 0.50)[0]]
    neg_pred = seq_pred[np.where(seq_pred <= 0.50)[0]]
       
    return neg_seq, neg_pred, seq_pred