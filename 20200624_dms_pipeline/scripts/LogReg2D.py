# Import libraries
import numpy as np
from Bio.Alphabet import IUPAC
from sklearn.linear_model import LogisticRegression

# Import custom functions
from utils import one_hot_encoder, plot_ROC_curve, \
    plot_PR_curve, calc_stat


def LogReg2D_classification(dataset, filename):
    """
    Classification of data with 2D logistic regression,
    followed by plotting of ROC and PR curves.

    Parameters
    ---
    dataset: the input dataset, containing training and
       test split data, and the corresponding labels
       for binding- and non-binding sequences.

    filename: an identifier to distinguish different
       plots from each other.

    Returns
    ---
    stats: array containing classification accuracy, precision
        and recall
    """

    # Import training/test set
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_test = dataset.test.loc[:, 'AASeq'].values

    # One hot encode the sequences in 2D
    X_train = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_train]
    X_train_2D_list = []
    for x in range(0, len(X_train)):
        X_train_2D = np.empty([20, 0])
        for y in range(0, X_train[x].shape[1]-1):
            for z in range(0, X_train[x].shape[0]):
                X_train_2D = np.concatenate(
                    (X_train_2D, X_train[x][z, y]*X_train[x][:, y+1:]), axis=1)
        X_train_2D_list.append(X_train_2D)
    X_train = [x.flatten('F') for x in X_train_2D_list]

    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test_2D_list = []
    for x in range(0, len(X_test)):
        X_test_2D = np.empty([20, 0])
        for y in range(0, X_test[x].shape[1]-1):
            for z in range(0, X_test[x].shape[0]):
                X_test_2D = np.concatenate(
                    (X_test_2D, X_test[x][z, y]*X_test[x][:, y+1:]), axis=1)
        X_test_2D_list.append(X_test_2D)
    X_test = [x.flatten('F') for x in X_test_2D_list]

    # Extract labels of training/test set
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values

    # Fitting Logistic Regression to the training set
    LR_classifier = LogisticRegression(random_state=0)
    LR_classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = LR_classifier.predict(X_test)
    y_score = LR_classifier.predict_proba(X_test)

    # ROC curve
    title = '2D Logistic Regression ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/2DLR_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = '2D Logistic Regression Precision-Recall curve (Train={})'.format(
        filename
    )
    plot_PR_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/2DLR_P-R_Test_{}.png'.format(filename)
    )

    # Calculate statistics
    stats = calc_stat(y_test, y_pred)

    # Return statistics
    return stats
