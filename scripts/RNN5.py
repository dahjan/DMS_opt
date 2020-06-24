# LSTM Recurrent Neural Network

def RNN_classification(dataset, ratio, filename):

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

    # One hot encode the sequences
    from utils import one_hot_encoder
    from Bio.Alphabet import IUPAC
    
    X_train = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_train]
    X_train = np.transpose(np.asarray(X_train), (0, 2, 1))

    X_test = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_test]
    X_test = np.transpose(np.asarray(X_test), (0, 2, 1))
    
    X_val = [one_hot_encoder(s = x, alphabet = IUPAC.protein) for x in X_val]
    X_val = np.transpose(np.asarray(X_val), (0, 2, 1))

    # Define the class labels
    y_train = dataset.train.loc[:, 'AgClass'].values
    y_test = dataset.test.loc[:, 'AgClass'].values
    y_val = dataset.val.loc[:, 'AgClass'].values
    
    # Building the RNN

    # Importing ther Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.callbacks import Callback
    
    class TimeHistory(Callback):
        def on_train_begin(self, logs={}):
            self.times = []
    
        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()
    
        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
            
    time_callback = TimeHistory()
     
    # Initializing the RNN
    RNN_classifier = Sequential()

    # Adding the first LSTM layer and some Dropout regularization
    RNN_classifier.add(LSTM(units = 40, return_sequences = True, input_shape = (10, 20)))
    RNN_classifier.add(Dropout(rate = 0.1)) #20% of nodes

    # Adding the second LSTM layer and some Dropout regularization
    RNN_classifier.add(LSTM(units = 40, return_sequences = True))
    RNN_classifier.add(Dropout(rate = 0.1)) #20% of nodes

    # Adding the third LSTM layer and some Dropout regularization
    RNN_classifier.add(LSTM(units = 40, return_sequences = False))
    RNN_classifier.add(Dropout(rate = 0.1)) #20% of nodes

    # Adding the output layer
    RNN_classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the RNN
    RNN_classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


    # Fitting the RNN to the Training set
    start_time = time.time()
    history = RNN_classifier.fit(x = X_train, y = y_train, shuffle = True, validation_data = (X_val, y_val), epochs = 20, batch_size = 32, callbacks=[time_callback])
    end_time = time.time()
    train_time = end_time - start_time
    
    # times = time_callback.times
    
    # Make prediction
    start_time = time.time()
    y_pred = RNN_classifier.predict(x = X_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    
    # Evaluate and plot model performance on test sets
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from inspect import signature
    from pylab import savefig
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('RNN Model Accuracy (Train={})'.format(filename))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    savefig('figures/RNN_Acc_{}.png'.format(filename))
    plt.cla()
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('RNN Model Loss (Train={})'.format(filename))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    savefig('figures/RNN_loss_{}.png'.format(filename))
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
    plt.title('RNN ROC curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/RNN_ROC_Test_{}.png'.format(filename))
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
    plt.title('RNN Precision-Recall curve (Train={})'.format(filename))
    plt.legend(loc = 'lower right')
    savefig('figures/RNN_P-R_Test_{}.png'.format(filename))
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
