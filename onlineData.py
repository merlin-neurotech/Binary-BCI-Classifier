#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
import myAnalysisTools as tools
import FeatureCalc as fc

def getData(classifier, mu_ft, std_ft):
    print('looking for an EEG stream...')
    streams = resolve_stream('type', 'EEG')
    if len(streams) ==0:
        raise(RunTimeError('Cant find EEG stream :('))
    window = 1   
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    descriptions = info.desc()
    sfreq = info.nominal_srate()
    n_samples = int(window * sfreq)
    n_chan = info.channel_count()
    print('Acquiring data...')
    data = np.zeros((n_samples, n_chan))
    times = np.arange(-window, 0, 1./sfreq)
    timer = time.time()
    while True:
        samples, timestamps = inlet.pull_chunk(timeout = 1.0, max_samples = 12)
        if timestamps:
            timestamps = np.float64(np.arange(len(timestamps))) #creates an array of numbers of numbers from 0 to length timestamps
            timestamps /= sfreq #divides that array by our sampling freq
            timestamps += times[-1] + 1/sfreq # not sure
            times = np.concatenate([times, timestamps])#adds timestamps to the end of the times array
            times = times[-n_samples:] #takes the last n_samples from times
            data = np.vstack([data, samples]) #adds our new samples to the data array
            data = data[-n_samples:]
            timer = time.time()
            n_samples, n_chan = data.shape
            epochs, remainder = tools.epoching(data, n_samples, samples_overlap = 0)
            feature_matrix = tools.compute_feature_matrix(epochs, sfreq)
            y_hat = tools.classifier_test(classifier, feature_matrix, mu_ft, std_ft)
            print(y_hat)