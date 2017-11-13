#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:40:40 2017

@author: omrinachmani
"""
import DataCollect
import myAnalysisTools as tools
import numpy as np
import onlineData

def bciRun():
    welcomeString = """Welcome to the BCI builder, we will start by recording some EEG data to train our algorithm
    There will be a concentrating phase and a resting phase. When concentrating, focus on something in your environment 
    and state at it for the duration of recording. When relaxing, just close your eyes and let your mind flow"""
    print("\n", welcomeString, "\n")
    
    userInput = input("When ready, press Enter to collect data for the concentrating phase:")
    if userInput == '':
        print("Please be still for data collection\n")
        data, sfreq = DataCollect.getData()
        epochs, remainder = tools.epoching(data, 100, samples_overlap = 0)
        feature_matrix_0 = tools.compute_feature_matrix(epochs, sfreq)
    
    print("Great! Now lets record while you are relaxing\n")
    userInput2 = input("Press Enter when you are ready: ")   
    if userInput2 == '':
        print("Please be still for data collection\n")
        data, sfreq = DataCollect.getData()
        epochs, remainder = tools.epoching(data, 100, samples_overlap = 0)
        feature_matrix_1 = tools.compute_feature_matrix(epochs, sfreq)
    print("\nWe are now ready to train the classifier!\n")
    userInput3 = input("\nPress enter when ready:")
    if userInput3 == '':
        print("Training Classifier...")
        classifier, mu_ft, std_ft = tools.classifier_train(feature_matrix_0, feature_matrix_1)
        print("Ok, lets test it...")
        onlineData.getData(classifier,mu_ft,std_ft)

    
bciRun()
        
        