#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:24:27 2017

@author: omrinachmani
"""
import numpy as np
import FeatureCalc as fc
from sklearn import svm

def epoching(data, samples_epoch, samples_overlap=0):
    n_samples, n_chan = data.shape
    
    samples_shift = samples_epoch - samples_overlap

    n_epochs =  int(np.floor( (n_samples - samples_epoch) / float(samples_shift) ) + 1 )

    #markers indicates where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0,n_epochs + 1)) * samples_shift;
    markers = markers.astype(int)
    #Divide data in epochs
    epochs = np.zeros((samples_epoch, n_chan, n_epochs));

    for i_epoch in range(0,n_epochs):
        epochs[:,:,i_epoch] = data[ markers[i_epoch] : markers[i_epoch] + samples_epoch ,:]
        
    if (markers[-1] != n_samples): 
        remainder = data[markers[-1] : n_samples, :]
    else:
        remainder = np.asarray([])
    
    return epochs , remainder

def compute_feature_matrix(epochs, Fs):
    """
    Call compute_feature_vector for each EEG epoch contained in the "epochs"
    
    """
    n_epochs = epochs.shape[2]    
        
    for i_epoch in range(n_epochs):
        
        if i_epoch == 0:
            feat = fc.compute_feature_vector(epochs[:,:,i_epoch], Fs).T
            feature_matrix = np.zeros((n_epochs, feat.shape[0])) # Initialize feature_matrix
            
        feature_matrix[i_epoch, :] = fc.compute_feature_vector(epochs[:,:,i_epoch], Fs).T 

    return feature_matrix

def classifier_train(feature_matrix_0, feature_matrix_1, algorithm = 'SVM'):
    """
    Trains a binary classifier using the SVM algorithm with the following parameters
    
    Arguments
    feature_matrix_0: Matrix with examples for Class 0
    feature_matrix_1: Matrix with examples for Class 1
    algorithm: Currently only SVM is supported
    
    Outputs
    classfier: trained classifier (scikit object)
    mu_ft, std_ft: normalization parameters for the data
    """
    # Create vector Y (class labels)
    class0 = np.zeros((feature_matrix_0.shape[0],1))
    class1 = np.ones((feature_matrix_1.shape[0],1))
    
    # Concatenate feature matrices and their respective labels
    y = np.ravel(np.concatenate((class0, class1),axis=0))
    features_all = np.concatenate((feature_matrix_0, feature_matrix_1),axis=0)
    
    # Normalize features, columnwise
    mu_ft = np.mean(features_all, axis=0)
    std_ft = np.std(features_all, axis=0)
    
    X = (features_all - mu_ft) / std_ft
    
    # Train SVM, using default parameters     
    classifier = svm.SVC()
    classifier.fit(X, y)
    
    return classifier, mu_ft, std_ft 

def classifier_test(classifier, feature_vector, mu_ft, std_ft):
    """
    Test the classifier on new data points.
    
    Arguments
    classifier: trained classifier (scikit object)
    feature_vector: np.array of shape [number of feature points; number of different features]
    mu_ft, std_ft: normalization parameters for the data
    
    Output
    y_hat: decision of the classifier on the data points
    """
    
    # Normalize feature_vector
    x = (feature_vector - mu_ft) / std_ft    
    y_hat = classifier.predict(x)
    #y_hat = None
    return y_hat
        
    
    