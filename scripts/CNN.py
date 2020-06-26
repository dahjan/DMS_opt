# Import libraries
import os
import numpy as np
from Bio.Alphabet import IUPAC
from keras.optimizers import Adam
from contextlib import redirect_stdout

# Import custom functions
from utils import one_hot_encoder, create_cnn, \
    plot_ROC_curve, plot_PR_curve, calc_stat


def CNN_classification(dataset, filename, save_model=False):
    """
    Classification of data with a convolutional neural
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

    # Set parameters for CNN
    params = [['CONV', 400, 3, 1],
              ['DROP', 0.5],
              ['POOL', 2, 1],
              ['FLAT'],
              ['DENSE', 50]]

    # Create the CNN with above-specified parameters
    CNN_classifier = create_cnn(params, 'relu', None)

    # Compiling the CNN
    opt = Adam(learning_rate=0.000075)
    CNN_classifier.compile(optimizer=opt, loss='binary_crossentropy',
                           metrics=['accuracy'])

    # Fit the CNN to the training set
    _ = CNN_classifier.fit(
        x=X_train, y=y_train, shuffle=True, validation_data=(X_val, y_val),
        epochs=20, batch_size=16, verbose=2
    )

    # Predicting the test set results
    y_pred = CNN_classifier.predict(x=X_test)

    # ROC curve
    title = 'CNN ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/CNN_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'CNN Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/CNN_P-R_Test_{}.png'.format(filename)
    )
    # Save model if specified
    if save_model:
        # Model summary
        with open(os.path.join(save_model, 'CNN_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                CNN_classifier.summary()

        # Model weights
        CNN_classifier.save(
            os.path.join(save_model, 'CNN_HER2')
        )

        # Return classification model
        return CNN_classifier
    else:
        # Probabilities larger than 0.5 are significant
        y_pred_stand = (y_pred > 0.5)

        # Calculate statistics
        stats = calc_stat(y_test, y_pred_stand)

        # Return statistics
        return stats
