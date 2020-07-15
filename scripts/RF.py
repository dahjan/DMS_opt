# Import libraries
from sklearn.ensemble import RandomForestClassifier

# Import custom functions
from utils import prepare_data, plot_ROC_curve, \
    plot_PR_curve, calc_stat


def RF_classification(dataset, filename):
    """
    Classification of data with random forests,
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
    RF_classifier = RandomForestClassifier(n_estimators=150)
    RF_classifier.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = RF_classifier.predict(X_test)
    y_score = RF_classifier.predict_proba(X_test)

    # ROC curve
    title = 'Random Forest ROC curve (Train={})'.format(filename)
    plot_ROC_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/RF_ROC_Test_{}.png'.format(filename)
    )

    # Precision-recall curve
    title = 'Random Forest Precision-Recall curve (Train={})'.format(filename)
    plot_PR_curve(
        y_test, y_score[:, 1], plot_title=title,
        plot_dir='figures/RF_P-R_Test_{}.png'.format(filename)
    )

    # Calculate statistics
    stats = calc_stat(y_test, y_pred)

    # Return statistics
    return stats
