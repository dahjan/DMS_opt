# Import libraries
from sklearn.linear_model import LogisticRegression

# Import custom functions
from utils import prepare_data, plot_ROC_curve, \
    plot_PR_curve, calc_stat


def LogReg_classification(dataset, filename):
    """
    Classification of data with logistic regression,
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

    # Fitting Logistic Regression to the training set
    LR_classifier = LogisticRegression(random_state=0)
    LR_classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = LR_classifier.predict(X_test)
    y_score = LR_classifier.predict_proba(X_test)

    # ROC curve
    title = 'Logistic Regression ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/LR_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'Logistic Regression Precision-Recall curve (Train={})'.format(
        filename
    )
    plot_PR_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/LR_P-R_Test_{}.png'.format(filename)
    )

    # Calculate statistics
    stats = calc_stat(y_test, y_pred)

    # Return statistics
    return stats
