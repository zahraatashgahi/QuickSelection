

"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
import numpy as np   
from sklearn import preprocessing  
import scipy
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split
import urllib.request as urllib2 
import errno
import os
import numpy as np
import sys; sys.path.append(os.getcwd())  

"""*************************************************************************"""
"""                             Load data                                   """
    
def load_data(name):

    if name == "coil20":
        mat = scipy.io.loadmat('./datasets/COIL20.mat')
        X = mat['fea']
        y = mat['gnd'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.StandardScaler().fit(X_train)
        
    elif name=="madelon":
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        X_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        X_test =  np.loadtxt(urllib2.urlopen(val_data_url))
        y_test =  np.loadtxt(urllib2.urlopen(val_resp_url))
        scaler = preprocessing.StandardScaler().fit(X_train)
        
    elif name=="isolet":
        import pandas as pd 
        data= pd.read_csv('./datasets/isolet.csv')
        data = data.values 
        X = data[:,:-1]
        X = X.astype("float")
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        for i in range(len(y_train)):
            y_train[i] = int(y_train[i][1])
        for i in range(len(y_test)):
            y_test[i] = int(y_test[i][1])
        y_train = y_train.astype("float")
        y_test = y_test.astype("float")
        scaler = preprocessing.StandardScaler().fit(X_train)
        
    elif name=="har":         
        X_train = np.loadtxt('./datasets/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('./datasets/UCI HAR Dataset/train/y_train.txt')
        X_test =  np.loadtxt('./datasets/UCI HAR Dataset/test/X_test.txt')
        y_test =  np.loadtxt('./datasets/UCI HAR Dataset/test/y_test.txt')
        scaler = preprocessing.StandardScaler().fit(X_train)
       
    elif name == "MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')    
        scaler = preprocessing.StandardScaler().fit(X_train)

    elif name=="SMK":
        mat = scipy.io.loadmat('./datasets/SMK_CAN_187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    
    elif name=="GLA":
        mat = scipy.io.loadmat('./datasets/GLA-BRA-180.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif name=="mac":
        mat = scipy.io.loadmat('./datasets/PCMAC.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    return X_train, y_train, X_test, y_test
    
def check_path(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise 





