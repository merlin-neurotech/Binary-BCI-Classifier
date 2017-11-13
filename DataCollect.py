#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:00:54 2017

@author: omrinachmani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 21:35:39 2017

@author: omrinachmani
"""
import numpy as np
from pylsl import StreamInlet, resolve_stream
import time

def getData(runtime=60):
    print('looking for an EEG stream...')
    runTime = runtime
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
    data = np.zeros((1, n_chan))
    times = np.arange(-window, 0, 1./sfreq)
    timer = time.time()
    end = timer + runTime
    while end > timer:
    
        samples, timestamps = inlet.pull_chunk(timeout = 1.0, max_samples = 12)
        if timestamps:
            timestamps = np.float64(np.arange(len(timestamps))) #creates an array of numbers of numbers from 0 to length timestamps
            timestamps /= sfreq #divides that array by our sampling freq
            timestamps += times[-1] + 1/sfreq # not sure
            times = np.concatenate([times, timestamps])#adds timestamps to the end of the times array
            times = times[-n_samples:] #takes the last n_samples from times
            data = np.vstack([data, samples]) #adds our new samples to the data array
            timer = time.time()

            
    data = np.delete(data, 0, 0) #deletes first column of zeros
    return data, sfreq
    
            
        
        