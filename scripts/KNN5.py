# Import libraries
from Bio.Alphabet import IUPAC
from sklearn.neighbors import KNeighborsClassifier

# Import custom functions
from utils import one_hot_encoder, plot_ROC_curve, \
    plot_PR_curve


def KNN_classification(dataset, filename):
    """
    Classification of data with k-nearest neighbors,
    followed by plotting of ROC and PR curves.

    Parameters
    ---
    dataset: the input dataset, containing training and
       test split data, and the corresponding labels
       for binding- and non-binding sequences.

    filename: an identifier to distinguish different
       plots from each other.
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
    X_test = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_test]
    X_test = [x.flatten('F') for x in X_test]
    X_val = [one_hot_encoder(s=x, alphabet=IUPAC.protein) for x in X_val]
    X_val = [x.flatten('F') for x in X_val]

    # Extract labels of training/test set
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values

    # Fitting classifier to the training set
    KNN_classifier = KNeighborsClassifier(
        n_neighbors=100, metric='minkowski', p=2)
    KNN_classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = KNN_classifier.predict(X_test)
    y_score = KNN_classifier.predict_proba(X_test)

    # ROC curve
    title = 'KNN ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/KNN_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'KNN Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/KNN_P-R_Test_{}.png'.format(filename)
    )

    return y_pred
