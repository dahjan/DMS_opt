# Import libraries
import os
import numpy as np
from Bio.Alphabet import IUPAC
from contextlib import redirect_stdout

# Import custom functions
from utils import one_hot_encoder, create_rnn, \
    plot_ROC_curve, plot_PR_curve, calc_stat


def RNN_classification(dataset, filename, save_model=False):
    """
    Classification of data with a recurrent neural
    network, followed by plotting of ROC and PR curves.

    Parameters
    ---
    dataset: the input dataset, containing training and
       test split data, and the corresponding labels
       for binding- and non-binding sequences.

    filename: an identifier to distinguish different
       plots from each other.

    save_model: optional; if provided, should specify the directory
       to save model summary and weights. The classification model
       will be returned in this case.
       If False, an array containing classification accuracy,
       precision and recall will be returned instead.
    """

    # Import training/test set
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_test = dataset.test.loc[:, 'AASeq'].values
    X_val = dataset.val.loc[:, 'AASeq'].values

    # One hot encode the sequences
    X_train = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_train]
    X_train = np.transpose(np.asarray(X_train), (0, 2, 1))
    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test = np.transpose(np.asarray(X_test), (0, 2, 1))
    X_val = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_val]
    X_val = np.transpose(np.asarray(X_val), (0, 2, 1))

    # Extract labels of training/test/validation set
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    y_val = dataset.val.loc[:, 'AgClass'].values

    # Building the RNN
    RNN_classifier = create_rnn()

    # Compiling the RNN
    RNN_classifier.compile(
        optimizer='rmsprop', loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Fit the RNN to the training set
    _ = RNN_classifier.fit(
        x=X_train, y=y_train, shuffle=True, validation_data=(X_val, y_val),
        epochs=20, batch_size=32, verbose=2
    )

    # Predicting the test set results
    y_pred = RNN_classifier.predict(x=X_test)

    # ROC curve
    title = 'RNN ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/RNN_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'RNN Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/RNN_P-R_Test_{}.png'.format(filename)
    )

    # Save model if specified
    if save_model:
        # Model summary
        with open(os.path.join(save_model, 'RNN_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                RNN_classifier.summary()

        # Model weights
        RNN_classifier.save(
            os.path.join(save_model, 'RNN_HER2')
        )

        # Return classification model
        return RNN_classifier
    else:
        # Probabilities larger than 0.5 are significant
        y_pred_stand = (y_pred > 0.5)

        # Calculate statistics
        stats = calc_stat(y_test, y_pred_stand)

        # Return statistics
        return stats
