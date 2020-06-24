# SVM (Linear Kernel)

def LSVM_classification(dataset, ratio, filename):

    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import time
    
    # Importing the dataset
    
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_train = [x[0:-1] for x in X_train]
    X_test_seq = dataset.test.loc[:, 'AASeq'].values
    X_test = [x[0:-1] for x in X_test_seq]
    X_val_seq = dataset.val.loc[:, 'AASeq'].values
    X_val = [x[0:-1] for x in X_val_seq]
    
    # One hot encode the sequence
    from scripts.utils import one_hot_encoder
    from Bio.Alphabet import IUPAC
    X_train = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_train]
    X_train = [x.flatten('F') for x in X_train]
    X_test = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_test]
    X_test = [x.flatten('F') for x in X_test]
    X_val = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_val]
    X_val = [x.flatten('F') for x in X_val]
    
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    
    """# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)"""
    
    # Fitting classifier to the Training set
    from sklearn.svm import LinearSVC
    SVM_classifier = LinearSVC()
    
    start_time = time.time()
    SVM_classifier.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    # Predicting the Test set results
    start_time = time.time()
    y_pred = SVM_classifier.predict(X_test)
    end_time = time.time()
    test_time = end_time - start_time
    

    # Evaluate and plot model performance on test sets
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from inspect import signature
    from pylab import savefig
    
    y_score = SVM_classifier.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = 'darkorange',
             lw=2, label = 'ROC curve (area - %0.2f)' % roc_auc)
    plt.plot([0, 1], [0,1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Linear SVM ROC curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/LSVM_ROC_Test_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    
    step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
    plt.step(recall, precision, color = 'navy', alpha = 0.2, where = 'post', label = 'Avg. Precision: {0:0.2f}'.format(average_precision))
    plt.fill_between(recall, precision, alpha = 0.2, color = 'navy', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Linear SVM Precision-Recall curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/LSVM_P-R_Test_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
   
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    acc = (tp+tn)/(tp+tn+fp+fn)    
    prec = (tp)/(tp+fp)
    recall = tp/(tp+fn)
    
    stats = np.array([acc, prec, recall, train_time, test_time])
    
    return y_pred, stats
    
