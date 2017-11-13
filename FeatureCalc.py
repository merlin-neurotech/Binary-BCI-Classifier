#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:11:00 2017

@author: omrinachmani
"""
import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, lfilter, lfilter_zi, firwin
from sklearn import svm

def compute_feature_vector(data, sfreq):
    n_samples, n_chan = data.shape
    w = np.hamming(n_samples)
    dataWinCentered = data - np.mean(data, axis =0)
    dataWinCenteredHam = (dataWinCentered.T*w).T    
    n = 1
    while n < n_samples: 
        n *= 2
    NFFT = n
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0)/n_samples
    PSD = 2*np.abs(Y[0:int(NFFT/2),:])
    f = sfreq/2*np.linspace(0,1,NFFT/2)   
    ind_delta, = np.where(f<4)
    meanDelta = np.mean(PSD[ind_delta,:],axis=0)
    # Theta 4-8
    ind_theta, = np.where((f>=4) & (f<=8))
    meanTheta = np.mean(PSD[ind_theta,:],axis=0)
    # Low alpha 8-10
    ind_alpha, = np.where((f>=8) & (f<=10)) 
    meanLowAlpha = np.mean(PSD[ind_alpha,:],axis=0)
    # Medium alpha
    ind_alpha, = np.where((f>=9) & (f<=11))
    meanMedAlpha = np.mean(PSD[ind_alpha,:],axis=0)
    # High alpha 10-12
    ind_alpha, = np.where((f>=10) & (f<=12)) 
    meanHighAlpha = np.mean(PSD[ind_alpha,:],axis=0)
    # Low beta 12-21
    ind_beta, = np.where((f>=12) & (f<=21))
    meanLowBeta = np.mean(PSD[ind_beta,:],axis=0)
    # High beta 21-30
    ind_beta, = np.where((f>=21) & (f<=30))
    meanHighBeta = np.mean(PSD[ind_beta,:],axis=0)
    # Alpha 8 - 12
    ind_alpha, = np.where((f>=8) & (f<=12))
    meanAlpha = np.mean(PSD[ind_alpha,:],axis=0)
    # Beta 12-30
    ind_beta, = np.where((f>=12) & (f<=30))
    meanBeta = np.mean(PSD[ind_beta,:],axis=0)
    feature_vector = np.concatenate((meanDelta, meanTheta, meanLowAlpha, meanHighAlpha, 
                                         meanLowBeta, meanHighBeta),axis=0) 
    return feature_vector