# Import libraries
from sklearn.svm import SVC

# Import custom functions
from utils import prepare_data, plot_ROC_curve, \
    plot_PR_curve, calc_stat


def SVM_classification(dataset, filename):
    """
    Classification of data with support vectors,
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

    # Import and one hot encode training/test set
    X_train, X_test, y_train, y_test = prepare_data(dataset)

    # Fitting classifier to the training set
    SVM_classifier = SVC(kernel='rbf')
    SVM_classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = SVM_classifier.predict(X_test)
    y_score = SVM_classifier.decision_function(X_test)

    # ROC curve
    title = 'SVM ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_score, plot_title=title,
        plot_dir='figures/SVM_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'SVM Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_score, plot_title=title,
        plot_dir='figures/SVM_P-R_Test_{}.png'.format(filename)
    )

    # Calculate statistics
    stats = calc_stat(y_test, y_pred)

    # Return statistics
    return stats
