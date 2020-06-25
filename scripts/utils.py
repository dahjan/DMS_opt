"""
Created on Wed Jan 23 08:52:23 2019

@author: dmason, frsimon
"""

# Importing libraries
import keras
import numpy as np
import pandas as pd
from pylab import savefig
from inspect import signature
from Bio.Alphabet import IUPAC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score


def mixcr_input(file_name, Ag_class):
    """
    Read in data from the MiXCR txt output file

    Parameters
    ---
    file_name: file name of the MiXCR txt file to read

    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1, non-binder = 0)

    """

    # Read data and rename columns
    x = pd.read_table(file_name)
    x = x[['Clone count', 'Clone fraction',
           'N. Seq. CDR3 ', 'AA. Seq.CDR3 ']]
    x = x.rename(index=str, columns={
        'Clone count': 'Count', 'Clone fraction': 'Fraction',
        'N. Seq. CDR3 ': 'NucSeq', 'AA. Seq.CDR3 ': 'AASeq'
    })

    # Select length and drop duplicate sequences
    x = x[(x.AASeq.str.len() == 15) & (x.Count > 1)]
    x = x.drop_duplicates(subset='AASeq')

    # Remove stop codons and incomplete codon sequences (*, _)
    idx = [i for i, aa in enumerate(x['AASeq']) if '*' not in aa]
    x = x.iloc[idx, :]
    idx = [i for i, aa in enumerate(x['AASeq']) if '_' not in aa]
    x = x.iloc[idx, :]

    if Ag_class == 0:
        x['AgClass'] = 0
    if Ag_class == 1:
        x['AgClass'] = 1

    return x


def data_split(Ag_pos, Ag_neg):
    """
    Create a collection of the data set and split into the
    training set and two test sets. One test set contains
    the same class split as the overall data set, the other
    test set contains a class split of approx. 10% binders
    and 90% non-binders.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set

    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Combine the positive and negative data frames
    Ag_combined = pd.concat([Ag_pos, Ag_neg])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

    # 70%/30% training test data split
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3
    )

    # 50%/50% training test data split
    idx2 = np.arange(0, idx_test.shape[0])
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5
    )

    # Create the test set with a 90%/10% non-binder/binder split
    Test_AgNeg_amt = len(Ag_combined.iloc[idx_test2, :]['AgClass']) - \
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'].sum()
    Adj_test_amt = int(Test_AgNeg_amt/0.90)
    Test_AgPos_amt = Adj_test_amt - Test_AgNeg_amt
    Test_AgPos = Ag_combined.iloc[idx_test, :].iloc[idx_test2, :].iloc[np.where(
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'] == 1)[0], :]
    Test_AgNeg = Ag_combined.iloc[idx_test, :].iloc[idx_test2, :].iloc[np.where(
        Ag_combined.iloc[idx_test, :].iloc[idx_test2, :]['AgClass'] == 0)[0], :]
    Updated_test = pd.concat([Test_AgNeg, Test_AgPos[0:Test_AgPos_amt]])

    # Create collection
    Seq_Ag_data = Collection(train=Ag_combined.iloc[idx_train, :],
                             val=Ag_combined.iloc[idx_test,
                                                  :].iloc[idx_val, :],
                             test=Updated_test,
                             complete=Ag_combined)

    return Seq_Ag_data


def data_split_adj(Ag_pos, Ag_neg, ratio):
    """
    Create a collection of the data set and split into the
    training set and two test sets. Data set is adjusted to
    the desired class split ratio. One test set contains the
    same class split as the overall data set, the other test
    set contains a class split of approx. 10% binders and
    90% non-binders.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    ratio: The desired class split ratio
    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Calculate data sizes based on ratio
    data_size_pos = len(Ag_pos)/ratio
    data_size_neg = len(Ag_neg)/(1-ratio)

    # Adjust the length of the data frames to meet the ratio requirement
    if len(Ag_pos) <= len(Ag_neg):
        if data_size_neg < data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*ratio))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((data_size_neg*ratio)):len(Ag_pos)]

        if data_size_neg >= data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(ratio)))]
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-ratio)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*ratio)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-ratio))):len(Ag_neg)]]
            )
    else:
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-ratio)))]
            Unused = Ag_pos[int((data_size_pos*ratio)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(ratio)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-ratio)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*ratio)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-ratio))):len(Ag_neg)]]
            )

    # Combine the positive and negative data frames
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

    # 70/30 training test data split
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3
    )

    # 50/50 training test data split
    idx2 = np.arange(0, idx_test.shape[0])
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5
    )

    # Create collection
    Seq_Ag_data = Collection(
        train=Ag_combined.iloc[idx_train, :],
        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
        test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],
        complete=Ag_combined
    )

    return Seq_Ag_data, Unused


def one_hot_encoder(s,  alphabet):
    """
    One hot encoding of a biological sequence.

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


# def one_hot_decoder(x, alphabet):
#     """
#     Decodes a one-hot encoding to a biological sequence
#
#     Parameters
#     ---
#     x: array, n_size_alphabet, n_length_string
#         Sequence as one-hot encoding
#     alphabet: Alphabet object, http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html
#
#     Example
#     ---
#     encoding = one_hot_encoder(sequence, IUPAC.unambiguous_dna)
#     one_hot_decoder(encoding, IUPAC.unambiguous_dna)
#
#     Returns
#     ---
#     s : str, decoded sequence
#     """
#
#     d = {a: i for i, a in enumerate(alphabet.letters)}
#     inv_d = {i: a for a, i in d.items()}
#     s = (''.join(str(inv_d[i]) for i in np.argmax(x, axis=0)))
#
#     return s


def create_ann():
    """
    Generate the ANN layers with Keras wrapper with hard-coded
    layers and activation functions.
    """

    # Initialize the ANN
    model = keras.Sequential()

    # Input layer and 1st hidden layer
    # Activation function: Rectifier; 140 input, 50 nodes
    model.add(
        keras.layers.Dense(units=70,
                           kernel_initializer='uniform',
                           activation='relu',
                           input_dim=200)
    )
    model.add(keras.layers.Dropout(rate=0.1))

    # 2nd hidden layer
    # Activation function: Rectifier; 50 nodes
    model.add(
        keras.layers.Dense(units=70,
                           kernel_initializer='uniform',
                           activation='relu')
    )
    model.add(keras.layers.Dropout(rate=0.1))

    # 3rd hidden layer
    # Activation function: Rectifier; 50 nodes
    model.add(
        keras.layers.Dense(units=70,
                           kernel_initializer='uniform',
                           activation='relu')
    )
    model.add(keras.layers.Dropout(rate=0.1))

    # Output layer
    # Activation function: Sigmoid
    model.add(
        keras.layers.Dense(units=1,
                           kernel_initializer='uniform',
                           activation='sigmoid')
    )

    return model


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

    # Initialize the CNN
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer((10, 20)))

    # Build network
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
            raise NotImplementedError('Layer type not implemented')

    # Output layer
    # Activation function: Sigmoid
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def create_rnn():
    """
    Generate the ANN layers with Keras wrapper with hard-coded
    layers and activation functions.
    """

    # Initialize the RNN
    model = keras.Sequential()

    # 1st LSTM layer and Dropout regularization
    model.add(keras.layers.LSTM(units=40,
                                return_sequences=True,
                                input_shape=(10, 20)))
    model.add(keras.layers.Dropout(rate=0.1))

    # 2nd LSTM layer and Dropout regularization
    model.add(keras.layers.LSTM(units=40,
                                return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.1))

    # 3rd LSTM layer and Dropout regularization
    model.add(keras.layers.LSTM(units=40,
                                return_sequences=False))
    model.add(keras.layers.Dropout(rate=0.1))

    # Output layer
    model.add(keras.layers.Dense(units=1,
                                 activation='sigmoid'))

    return model


def plot_ROC_curve(y_test, y_score, plot_title, plot_dir):
    """
    Plots a Receiver Operating Characteristic (ROC) curve in
    the specified output directory.

    Parameters
    ---
    y_test: the labels for binding- and non-binding sequences.
    y_score: the predicted probabilities from the classifier.
    plot_title: the title of the plot.
    plot_dir: the directory for plotting.
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area - %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc='lower right')
    savefig(plot_dir)
    plt.cla()
    plt.clf()


def plot_PR_curve(y_test, y_score, plot_title, plot_dir):
    """
    Plots a precision-recall (PR) curve in the specified
    output directory.

    Parameters
    ---
    y_test: the labels for binding- and non-binding sequences.
    y_score: the predicted probabilities from the classifier.
    plot_title: the title of the plot.
    plot_dir: the directory and filename for plotting.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_test, y_score
    )
    average_precision = average_precision_score(y_test, y_score)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='navy', alpha=0.2, where='post',
             label='Avg. Precision: {0:0.2f}'.format(average_precision))
    plt.fill_between(recall, precision, alpha=0.2, color='navy', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(plot_title)
    plt.legend(loc='lower right')
    savefig(plot_dir)
    plt.cla()
    plt.clf()


# TODO: Continue here in checking the function!!

def seq_classification(classifier):
    """
    In silico generate sequences and classify them as a binding or non-binding
    sequence respectively.

    Parameters
    ---
    AA_per_pos: amino acids used per position in list format, i.e.,
    [['F','Y','W'],['A','D','G',...'Y']]

    classifier: RNN or CNN classification model to use

    """

    # Define valid amino acids per position
    AA_per_pos = [
        ['Y', 'W'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'D', 'F', 'G', 'H', 'I', 'L', 'N',
         'P', 'R', 'S', 'T', 'V', 'Y'],
        ['A', 'G', 'H', 'S', ],
        ['F', 'L'],
        ['Y'],
        ['A', 'E', 'K', 'L', 'P', 'Q', 'T', 'V'],
        ['F', 'H', 'I', 'L', 'N', 'Y'],
        ['A', 'D', 'E', 'H', 'I', 'K', 'L', 'M',
         'N', 'P', 'Q', 'T', 'V']]

    dim = []
    for x in AA_per_pos:
        # Get number of possible amino acids
        dim.append(len(x))

        idx = [0]*len(dim)
        print(idx)
        current_seq = np.empty(0, dtype=object)
        pos_seq = np.empty(0, dtype=str)
        pos_pred = np.empty(0, dtype=float)

    print(schnuebl)
    counter = 1
    pos = 0

    while(1):
        l = []
        for i in range(0, len(dim)):
            l.append(AA_per_pos[i][idx[i]])

        current_seq = np.append(current_seq, (''.join(l)))

        if len(current_seq) == 500:
            ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein)
                       for x in current_seq]
            ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
            seq_pred = classifier.predict(x=ohe_seq)

            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]])
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]])
            # pos_seq.append(current_seq[np.where(seq_pred > 0.5)[0]])

            current_seq = np.empty(0, dtype=object)
            print(counter, '/', np.prod(dim)/500)
            counter += 1
            # current_seq = []

        if sum(idx) == (sum(dim)-len(dim)):
            ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein)
                       for x in current_seq]
            ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
            seq_pred = classifier.predict(x=ohe_seq)

            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]])
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]])
            # pos_seq.append(current_seq[np.where(seq_pred > 0.5)[0]])

            break

        while(1):
            if (idx[pos]+1) == dim[pos]:
                idx[pos] = 0
                pos += 1
            else:
                idx[pos] += 1
                pos = 0
                break

    return pos_seq, pos_pred


def seq_class_test(AA_list, classifier):

    aa_test = [x[3:-2] for x in AA_list]

    ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in aa_test]
    ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
    seq_pred = classifier.predict(x=ohe_seq)

    pos_seq = AA_list[np.where(seq_pred > 0.50)[0]]
    pos_pred = seq_pred[np.where(seq_pred > 0.50)[0]]

    return pos_seq, pos_pred, seq_pred


def seq_class_test2(AA_list, classifier):

    aa_test = [x[3:-2] for x in AA_list]

    ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in aa_test]
    ohe_seq = [x.flatten('F') for x in ohe_seq]
    seq_pred = classifier.predict(ohe_seq)

    return seq_pred


def seq_class_test3(AA_list, classifier):

    aa_test = [x[3:-2] for x in AA_list]

    ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in aa_test]
    ohe_seq = [x.flatten('F') for x in ohe_seq]
    ohe_seq = np.asarray(ohe_seq)
    seq_pred = classifier.predict(ohe_seq)

    return seq_pred


def seq_class_neg(AA_list, classifier):

    aa_test = [x[3:-2] for x in AA_list]

    ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in aa_test]
    ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))
    seq_pred = classifier.predict(x=ohe_seq)

    neg_seq = AA_list[np.where(seq_pred <= 0.50)[0]]
    neg_pred = seq_pred[np.where(seq_pred <= 0.50)[0]]

    return neg_seq, neg_pred, seq_pred
