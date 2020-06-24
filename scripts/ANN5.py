# Artificial Neural Network

def ANN_classification(dataset, ratio, filename):

    # Data Preprocessing

    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import time

    # Split the data set into training, test1 (train split), and test2 (10/90)
    # Trim off 5' CSR and 3' YW amino acids
    X_train = dataset.train.loc[:, 'AASeq'].values
    X_train = [x[0:-1] for x in X_train]
    X_test_seq = dataset.test.loc[:, 'AASeq'].values
    X_test = [x[0:-1] for x in X_test_seq]
    X_val_seq = dataset.val.loc[:, 'AASeq'].values
    X_val = [x[0:-1] for x in X_val_seq]

    # One hot encode the sequence
    from utils import one_hot_encoder
    from Bio.Alphabet import IUPAC
    X_train = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_train]
    X_train = [x.flatten('F') for x in X_train]
    X_train = np.asarray(X_train)
    X_test = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_test]
    X_test = [x.flatten('F') for x in X_test]
    X_test = np.asarray(X_test)
    X_val = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_val]
    X_val = [x.flatten('F') for x in X_val]
    X_val = np.asarray(X_val)
    
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    y_val = dataset.val.loc[:, 'AgClass'].values
    
    # Building the ANN

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.callbacks import Callback
    from keras.optimizers import Adam
    
    ada_optimizer = Adam(learning_rate=0.0001)
    
    class TimeHistory(Callback):
        def on_train_begin(self, logs={}):
            self.times = []
    
        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()
    
        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
            
    time_callback = TimeHistory()
        
    # Initializing the ANN
    ANN_classifier = Sequential()
    
    # Adding the input layer and the first hidden layer (activation function: Rectifier; 140 input, 50 nodes)
    ANN_classifier.add(Dense(units = 70, kernel_initializer = 'uniform', activation = 'relu', input_dim = 200))
    ANN_classifier.add(Dropout(rate = 0.1))
    
    # Adding the second hidden layer (activation function: Rectifier; 50 nodes)
    ANN_classifier.add(Dense(units = 70, kernel_initializer = 'uniform', activation = 'relu'))
    ANN_classifier.add(Dropout(rate = 0.1))
    
    # Adding the third hidden layer (activation function: Rectifier; 50 nodes)
    ANN_classifier.add(Dense(units = 70, kernel_initializer = 'uniform', activation = 'relu'))
    ANN_classifier.add(Dropout(rate = 0.1))
    
    # Adding the output layer (activation function: Sigmoid)
    ANN_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    # adam = stochastic gradient descent; binary_crossentropy = logarithmic loss
    ANN_classifier.compile(optimizer = ada_optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])    
    
    # Fitting the ANN to the Training set
    start_time = time.time()
    history = ANN_classifier.fit(x = X_train, y = y_train, shuffle = True, validation_data = (X_val, y_val), batch_size = 16, epochs = 20, callbacks=[time_callback])
    end_time = time.time()
    train_time = end_time - start_time
    
    # times = time_callback.times
    
    start_time = time.time()
    y_pred = ANN_classifier.predict(x = X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    
    # Evaluate and plot model performance on test sets
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from inspect import signature
    from pylab import savefig
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy (Train={})'.format(filename))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    savefig('figures/ANN_Acc_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (Train={})'.format(filename))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    savefig('figures/ANN_loss_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    
    # ROC curve on test2
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = 'darkorange',
             lw=2, label = 'ROC curve (area - %0.2f)' % roc_auc)
    plt.plot([0, 1], [0,1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ANN ROC curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/ANN_ROC_test_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    
    # Precision-recall curve on test1
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    
    step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
    plt.step(recall, precision, color = 'navy', alpha = 0.2, where = 'post', label = 'Avg. Precision: {0:0.2f}'.format(average_precision))
    plt.fill_between(recall, precision, alpha = 0.2, color = 'navy', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ANN Precision-Recall curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/ANN_P-R_test_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    from sklearn.metrics import confusion_matrix
    y_pred_stand = (y_pred > 0.5)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_stand).ravel()
    
    acc = (tp+tn)/(tp+tn+fp+fn)    
    prec = (tp)/(tp+fp)
    recall = tp/(tp+fn)
    
    stats = np.array([acc, prec, recall, train_time, test_time])
    
    return y_pred, stats
