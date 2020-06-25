# Import libraries
import os
import numpy as np
from Bio.Alphabet import IUPAC
from keras.optimizers import Adam

# Import custom functions
from utils import one_hot_encoder, create_ann, \
    plot_ROC_curve, plot_PR_curve


def ANN_classification(dataset, filename, save_model=False):
    """
    Classification of data with an artificial neural
    network, followed by plotting of ROC and PR curves.

    Parameters
    ---
    dataset: the input dataset, containing training and
       test split data, and the corresponding labels
       for binding- and non-binding sequences.

    filename: an identifier to distinguish different
       plots from each other.

    save_model: optional; if provided, should specify the
       directory to save model summary and weights.

    Returns: ROC and PR curves.
    """

    # Import training/test set
    # Trim off 3' Y amino acid
    X_train_seq = dataset.train.loc[:, 'AASeq'].values
    X_train = [x[0:-1] for x in X_train_seq]
    X_test_seq = dataset.test.loc[:, 'AASeq'].values
    X_test = [x[0:-1] for x in X_test_seq]
    X_val_seq = dataset.val.loc[:, 'AASeq'].values
    X_val = [x[0:-1] for x in X_val_seq]

    # One hot encode the sequences
    X_train = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_train]
    X_train = [x.flatten('F') for x in X_train]
    X_train = np.asarray(X_train)
    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test = [x.flatten('F') for x in X_test]
    X_test = np.asarray(X_test)
    X_val = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_val]
    X_val = [x.flatten('F') for x in X_val]
    X_val = np.asarray(X_val)

    # Extract labels of training/test/validation set
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    y_val = dataset.val.loc[:, 'AgClass'].values

    # Create the ANN
    ANN_classifier = create_ann()

    # Compiling the ANN
    ada_optimizer = Adam(learning_rate=0.0001)
    ANN_classifier.compile(
        optimizer=ada_optimizer, loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Fit the ANN to the training set
    _ = ANN_classifier.fit(
        x=X_train, y=y_train, shuffle=True, validation_data=(X_val, y_val),
        batch_size=16, epochs=20, verbose=2
    )

    # Save model if specified
    if save_model:
        # Model summary
        with open(os.path.join(save_model, 'ANN_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                ANN_classifier.summary()

        # Model weights
        ANN_classifier.save(
            os.path.join(save_model, 'ANN_HER2')
        )

    # Predicting the test set results
    y_pred = ANN_classifier.predict(x=X_test)

    # ROC curve
    title = 'ANN ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/ANN_ROC_test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'ANN Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_pred, plot_title=title,
        plot_dir='figures/ANN_P-R_test_{}.png'.format(filename)
    )

    # Return flattened y_pred
    return list(np.concatenate(y_pred).flat)
