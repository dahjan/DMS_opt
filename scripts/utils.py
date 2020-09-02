"""
Created on Wed Jan 23 08:52:23 2019

@author: dmason, frsimon
"""

# Importing libraries
import sys
import keras
import numpy as np
import pandas as pd
from pylab import savefig
from inspect import signature
from Bio.Alphabet import IUPAC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix, f1_score, \
    matthews_corrcoef


def load_input_data(filenames, Ag_class):
    """
    Load the files specified in filenames.

    Parameters
    ---
    filenames: a list of names that specify the files to
        be loaded.

    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1, non-binder = 0)
    """

    # Combine the non-binding sequence data sets.
    # Non-binding data sets include Ab+ data and Ag-
    # sorted data for all 3 libraries
    l_data = []
    for file in filenames:
        l_data.append(
            mixcr_input('data/' + file, Ag_class, seq_len=15)
        )
    mHER_H3 = pd.concat(l_data)

    # Drop duplicate sequences
    mHER_H3 = mHER_H3.drop_duplicates(subset='AASeq')

    # Remove 'CAR/CSR' motif and last two amino acids
    mHER_H3['AASeq'] = [x[3:-2] for x in mHER_H3['AASeq']]

    # Shuffle sequences and reset index
    mHER_H3 = mHER_H3.sample(frac=1).reset_index(drop=True)

    return mHER_H3


def load_input_data_L3(filenames, Ag_class):
    """
    Load the files specified in filenames.

    Parameters
    ---
    filenames: a list of names that specify the files to
        be loaded.

    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1, non-binder = 0)
    """

    # Combine the non-binding sequence data sets.
    # Non-binding data sets include Ab+ data and Ag-
    # sorted data for all 3 libraries
    l_data = []
    for file in filenames:
        l_data.append(
            mixcr_input('data/' + file, Ag_class, seq_len=11)
        )
    mHER_L3 = pd.concat(l_data)

    # Drop duplicate sequences
    mHER_L3 = mHER_L3.drop_duplicates(subset='AASeq')

    # Remove first and last amino acids
    mHER_L3['AASeq'] = [x[1:-1] for x in mHER_L3['AASeq']]

    # Shuffle sequences and reset index
    mHER_L3 = mHER_L3.sample(frac=1).reset_index(drop=True)

    return mHER_L3


def mixcr_input(file_name, Ag_class, seq_len):
    """
    Read in data from the MiXCR txt output file

    Parameters
    ---
    file_name: file name of the MiXCR txt file to read

    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1, non-binder = 0)

    seq_len: the length of sequences; other lengths will be
             removed.
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
    x = x[(x.AASeq.str.len() == seq_len) & (x.Count > 1)]
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


def prepare_data(dataset):
    """
    Extract training and test data with the corresponding labels from
    the input dataset collection. The data is then one hot encoded.

    Parameters
    ---
    dataset: A collection of Ag+ and Ag- data, separated
        into training, test and validation set.

    Returns
    ---
    X_train, X_test, y_train, y_test: one hot encoded trainin and
        test data with the corresponding labels.
    """

    # Import training/test set
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_test = dataset.test.loc[:, 'AASeq'].values

    # One hot encode the sequences
    X_train = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_train]
    X_train = [x.flatten('F') for x in X_train]
    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test = [x.flatten('F') for x in X_test]

    # Extract labels of training/test set
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values

    return X_train, X_test, y_train, y_test


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

    Returns
    ---
    Seq_Ag_data: A collection of Ag+ and Ag- data, separated
        into training, test and validation set.
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

    # 50%/50% test validation data split
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


def data_split_L3(Ag_pos, Ag_neg):
    """
    Create a collection of the data set and split into
    training and test sets. The training set contains
    70%, the test set 30% of all data. The larger class
    in the data set is subsampled, so that there are an
    equal number of Ag, and Ag- sequences.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set

    Ag_neg: Dataframe of the Ag- data set

    fraction: The desired fraction of Ag+ in the data set
    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Combine the positive and negative data frames
    Ag_combined = pd.concat([Ag_pos, Ag_neg])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

    print("Class distribution before subsampling of larger class:")
    print(Ag_combined.AgClass.value_counts())

    # Get number of binding and non-binding sequences
    non_dupl = Ag_combined.AgClass.value_counts()
    size_neg = non_dupl[0]
    size_pos = non_dupl[1]

    # Subsample the larger class
    if size_pos <= size_neg:
        neg_idx = np.where(Ag_combined.AgClass == 0)
        rm_idx = np.random.choice(
            neg_idx[0], size=size_neg-size_pos, replace=False
        )
    else:
        pos_idx = np.where(Ag_combined.AgClass == 1)
        rm_idx = np.random.choice(
            pos_idx[0], size=size_pos-size_neg, replace=False
        )

    # Drop sequences of larger class from dataset
    Ag_combined1 = Ag_combined.drop(rm_idx)

    print("Class distribution after subsampling of larger class:")
    print(Ag_combined1.AgClass.value_counts())

    # 70%/30% training test data split
    idx = np.arange(0, Ag_combined1.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined1['AgClass'], test_size=0.3
    )

    # Create collection
    Seq_Ag_data = Collection(
        train=Ag_combined1.iloc[idx_train, :],
        test=Ag_combined1.iloc[idx_test, :],
        complete=Ag_combined1
    )

    return Seq_Ag_data


def data_split_adj(Ag_pos, Ag_neg, fraction):
    """
    Create a collection of the data set and split into the
    training set and two test sets. Data set is adjusted to
    match the specified class split fraction, which determines
    the fraction of Ag+ sequences.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    fraction: The desired fraction of Ag+ in the data set
    """

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # Calculate data sizes based on ratio
    data_size_pos = len(Ag_pos)/fraction
    data_size_neg = len(Ag_neg)/(1-fraction)

    # Adjust the length of the data frames to meet the ratio requirement
    if len(Ag_pos) <= len(Ag_neg):
        if data_size_neg < data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*fraction))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)]

        if data_size_neg >= data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-fraction))):len(Ag_neg)]]
            )
    else:
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-fraction)))]
            Unused = Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-fraction))):len(Ag_neg)]]
            )

    # Combine the positive and negative data frames
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1).reset_index(drop=True)

    # 70%/30% training test data split
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.3
    )

    # 50%/50% test validation data split
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
    alphabet: Alphabet object, downloaded from
        http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

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
    alphabet: Alphabet object, downloaded from
        http://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC-module.html

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


def calc_stat(y_test, y_pred):
    """
    Compute accuracy of the classification via a confusion matrix.

    Parameters
    ---
    y_test: the labels for binding- and non-binding sequences.
    y_pred: the classification of samples in the data X.

    Returns
    ---
    stats: array containing classification accuracy, precision
        and recall
    """

    # Calculate accuracy of classification via confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred
    ).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = (tp)/(tp+fp)
    recall = tp/(tp+fn)

    # Calculate F1 and MCC score
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Return statistics
    return np.array([acc, prec, recall, f1, mcc])


def create_ann():
    """
    Generate the ANN layers with a Keras wrapper, using hard-coded
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


def create_cnn(units_per_layer, input_shape,
               activation, regularizer):
    """
    Generate the CNN layers with a Keras wrapper.

    Parameters
    ---
    units_per_layer: architecture features in list format, i.e.:
        Filter information: [CONV, # filters, kernel size, stride]
        Max Pool information: [POOL, pool size, stride]
        Dropout information: [DROP, dropout rate]
        Flatten: [FLAT]
        Dense layer: [DENSE, number nodes]

    input_shape: a tuple defining the input shape of the data

    activation: Activation function, i.e. ReLU, softmax

    regularizer: Kernel and bias regularizer in convulational and dense
        layers, i.e., regularizers.l1(0.01)
    """

    # Initialize the CNN
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(input_shape))

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


def build_classifier(filters, kernels, strides, activation,
                     dropout, dense):
    """
    This function builds a CNN classifier, whose hyperparameters
    can be optimized with sklearn. Those parameters are the input
    of this function:
        filters, kernels, strides, activation,
        dropout, dense, learn_rate

    The function then returns the compiled CNN classifier.
    """

    # Initialize the classifier
    classifier = keras.Sequential()

    # 1. Convolutional layer
    classifier.add(keras.layers.Conv1D(filters=filters,
                                       kernel_size=kernels,
                                       strides=strides,
                                       activation=activation,
                                       padding='same',
                                       input_shape=(10, 20, )))

    # 2. Add dropout
    classifier.add(keras.layers.Dropout(rate=dropout))

    # 3. Max Pooling
    classifier.add(keras.layers.MaxPool1D(pool_size=2,
                                          strides=strides))

    # 4. Flatten the input
    classifier.add(keras.layers.Flatten())

    # 5. Dense layer
    classifier.add(keras.layers.Dense(units=dense,
                                      activation=activation))

    # 6. Output layer, sigmoid activation
    classifier.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compiling the classifier
    classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


def create_rnn():
    """
    Generate the RNN layers with a Keras wrapper, and hard-coded
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


def seq_classification(classifier, flatten_input=False):
    """
    In silico generate sequences and classify them as a binding
    or non-binding sequence respectively.
    Parameters
    ---
    classifier: The neural network classification model to use.

    flatten_input: If set True, the in silico generated sequence
        input is flattened before being classified. This is
        necessary for neural networks which take a 2-dimensional
        input (i.e. ANN).

    Returns
    ---
    pos_seq, pos_pred: an array of all positive sequences together
       with their predicted value.
    """

    # Define valid amino acids per position
    AA_per_pos = [
        ['F', 'Y', 'W'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V'],
        ['A', 'C', 'D', 'F', 'G', 'H', 'I', 'L',
         'N', 'P', 'R', 'S', 'T', 'V', 'Y'],
        ['A', 'G', 'S'],
        ['F', 'L', 'M'],
        ['Y'],
        ['A', 'E', 'K', 'L', 'M', 'P', 'Q', 'T', 'V'],
        ['F', 'H', 'I', 'L', 'N', 'Y'],
        ['A', 'D', 'E', 'H', 'I', 'K', 'L', 'M',
         'N', 'P', 'Q', 'T', 'V']]

    # Parameters used in while loop
    current_seq = np.empty(0, dtype=object)
    dim = [len(x) for x in AA_per_pos]
    idx = [0]*len(dim)
    counter = 1
    pos = 0

    # Arrays to store results
    pos_seq = np.empty(0, dtype=str)
    pos_pred = np.empty(0, dtype=float)

    while(1):
        # Get every possible combination of amino acids
        l_comb = []
        for i in range(0, len(dim)):
            l_comb.append(AA_per_pos[i][idx[i]])

        # Append current sequence
        current_seq = np.append(current_seq, (''.join(l_comb)))

        # Run classification on 500 sequences
        if len(current_seq) == 500:
            # Run classification of current_seq
            seq_pred = run_classification(
                current_seq, classifier, flatten_input
            )

            # Append significant sequences and predictions
            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]]
            )
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]]
            )

            # Empty current_seq array
            current_seq = np.empty(0, dtype=object)

            # Print progress bar
            progbar(counter, np.ceil(np.prod(dim)/500))
            counter += 1

        # Terminating condition
        if sum(idx) == (sum(dim)-len(dim)):
            # Run classification of current_seq
            seq_pred = run_classification(
                current_seq, classifier, flatten_input
            )

            # Append significant sequences and predictions
            pos_seq = np.append(
                pos_seq, current_seq[np.where(seq_pred > 0.50)[0]]
            )
            pos_pred = np.append(
                pos_pred, seq_pred[np.where(seq_pred > 0.50)[0]]
            )

            break

        # Update idx
        while(1):
            if (idx[pos]+1) == dim[pos]:
                idx[pos] = 0
                pos += 1
            else:
                idx[pos] += 1
                pos = 0
                break

    return pos_seq, pos_pred


def run_classification(current_seq, classifier, flatten_input):
    """
    Performs one hot encoding on a list of sequences; those
    sequences are the classified with a neural network model.

    Parameters
    ---
    current_seq: a numpy array containing the sequences
        to be classified.

    classifier: The neural network classification model to use.

    flatten_input: If set True, the in silico generated sequence
        input is flattened before being classified. This is
        necessary for neural networks which take a 2-dimensional
        input (i.e. ANN).

    Returns
    ---
    seq_pred: an array of prediction values for the sequences.
    """

    # One-hot encoding
    ohe_seq = [one_hot_encoder(s=x, alphabet=IUPAC.protein)
               for x in current_seq]
    ohe_seq = np.transpose(np.asarray(ohe_seq), (0, 2, 1))

    # Flatten the input if specified
    if flatten_input:
        ohe_seq = ohe_seq.reshape(ohe_seq.shape[0], -1)

    # Sequence classification
    seq_pred = classifier.predict(x=ohe_seq)

    return seq_pred


def progbar(i, iter_per_epoch, message='', bar_length=50):
    """
    Progress bar, written by Simon Friedensohn.

    Prints a progress bar in the following form:
        [==                       ] 8%
    """

    # Calculate current progress
    j = (i % iter_per_epoch) + 1

    # Create and print the progress bar
    perc = int(100. * j / iter_per_epoch)
    prog = ''.join(['='] * (bar_length * perc // 100))
    template = "\r[{:" + str(bar_length) + "s}] {:3d}% {:s}"
    string = template.format(prog, perc, message)
    sys.stdout.write(string)
    sys.stdout.flush()

    # Terminating condition
    end_epoch = (j == iter_per_epoch)
    if end_epoch:
        prog = ''.join(['='] * (bar_length))
        string = template.format(prog, 100, message)
        sys.stdout.write(string)
        sys.stdout.flush()
