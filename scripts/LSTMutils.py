# -*- coding: utf-8 -*-
"""
Created on 29 Sep 2019

Modules used in the other scripts related to data pre-processing, data preparation, and scoring.


@author: Achut Manandhar
Adapted from the following example:
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
"""
from numpy import arange
from numpy import digitize
from numpy import array
from numpy import concatenate
from numpy import split
from numpy import mean
from numpy import std
from sklearn.preprocessing import StandardScaler

# Bin labels
def waterColBinFun(waterColY):
    bins = arange(0,25,.1)
    inds = digitize(waterColY, bins)
    Ybin = inds/10
    return Ybin

# split a univariate dataset into train/test sets
def split_dataset(data,NdaysTrain,NhrBinsPerDay):
    # split into non-standard H-hour days
    train, test = data[:NdaysTrain*NhrBinsPerDay], data[NdaysTrain*NhrBinsPerDay:]    
    
    # restructure into windows of daily data
    train = array(split(train, len(train)/NhrBinsPerDay))
    test = array(split(test, len(test)/NhrBinsPerDay))
    return train, test

# Standarsize dataset
def standardize_dataset(train,test,NhrBinsPerDay):
    # flatten data
    train = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    test = test.reshape((test.shape[0]*test.shape[1], test.shape[2]))
    
    # scale features and labels separately
    train = array(train)
    test = array(test)
    trainY = train[:,-1].reshape((train.shape[0],1))
    testY = test[:,-1].reshape((test.shape[0],1))
    trainX = train[:,:-1]
    testX = test[:,:-1]    
    # Zmuv
    zmuv = StandardScaler()
    zmuv = zmuv.fit(trainX)
    trainX = zmuv.transform(trainX)
    testX = zmuv.transform(testX)
    # Zmuv
    zmuvY = StandardScaler()
    zmuvY = zmuvY.fit(trainY)
    trainY = zmuvY.transform(trainY)
    testY = zmuvY.transform(testY)
    # Concatenate labels
    train = concatenate((trainX,trainY),axis=1)
    test = concatenate((testX,testY),axis=1)
    
    # Reshape
    # restructure into windows of daily data
    train = array(split(train, len(train)/NhrBinsPerDay))
    test = array(split(test, len(test)/NhrBinsPerDay))
    
    return train, test, zmuv, zmuvY

# Transform data into a format required for LSTM dataset
def to_supervised(train, n_input):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		# ensure we have enough data for this instance
		if in_end < len(data):
        # Use all features
			x_input = data[in_start:in_end, :-1]
			X.append(x_input)
			y.append(data[in_end-1, -1])
		# move along one time step
		in_start += 1
	return array(X), array(y)
    
# Score 
def score_model(actual, predicted):
    # Absolute difference in metres                
    absDiff = abs(actual-predicted)
    score_mean = round(mean(absDiff),2)
    # One standard deviation in metres                
    score_std = round(std(absDiff),2) 
    return score_mean, score_std

