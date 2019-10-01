# -*- coding: utf-8 -*-
"""
Created on 29 Sep 2019


@author: Achut Manandhar
Adapted from the following example:
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import L1L2
from keras.optimizers import Adam
from numpy import arange
from numpy import array

class LSTMmultiInUniOut(object):
    '''
    Implements LSTM in Keras with tensorFlow backend
    Adapted from the example described in here:
    
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    '''
    def __init__(self, n_input = 3, n_hid = 50, dropout = 0.2, validation_split = .2, loss_fn = 'mse', optimizer = 'Adam', learn_rate = .0001, L1L2 = [0.0,0.0001], batch_size = 10, n_epoch = 140, verbose = 1, shuffle = 0):
        '''
        Initializes parameters for stacked denoising autoencoders
        @param n_input: number of layers, i.e., number of autoencoders to stack on top of each other.

        '''
        self.n_input = n_input                
        self.n_hid = n_hid        
        self.dropout = dropout        
        self.validation_split = validation_split        
        self.loss_fn = loss_fn        
        self.optimizer = optimizer        
        self.L1L2 = L1L2        
        self.learn_rate = learn_rate        
        self.batch_size = batch_size        
        self.n_epoch = n_epoch        
        self.verbose = verbose        
        self.shuffle = shuffle
        
    # Build model
    def build_model(self, train_x, train_y):            
        n_timesteps, n_features = train_x.shape[1], train_x.shape[2]
        # define model
    #    # One hidden layer with 50 units
        model = Sequential()    
        model.add(LSTM(self.n_hid, activation='relu', 
                       input_shape=(n_timesteps, n_features),                    
                       bias_regularizer = L1L2(l1=self.L1L2[0], l2=self.L1L2[1]),
                       kernel_regularizer = L1L2(l1=self.L1L2[0], l2=self.L1L2[1])))	
        model.add(Dropout(rate=self.dropout))	
        model.add(Dense(1))
        optimizer = self.optimizer
        model.compile(loss=self.loss_fn, optimizer=optimizer(lr=self.learn_rate))
    #    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=esMinDelta, verbose=1, patience=10)                      
        return model
    
    # Train model
    def train_model(self, model, train_x, train_y):
        # fit model                
        modelHistory = model.fit(train_x, train_y, 
                                 epochs=self.n_epoch, 
                                 batch_size=self.batch_size, 
                                 verbose=self.verbose, 
                                 shuffle=self.shuffle,
                                 validation_split=self.validation_split)
    #                             callbacks=[es]) 
        return modelHistory, model    
    
    
    def predict_model(self, model, train, test, NhrBinsPerDay):
        history = [x for x in train]
        # walk-forward validation over each day
        predictions = list()
        for i in range(len(test)):
            # get real observation and add to history for predicting this day
            history.append(test[i, :])        
            # predict the day
            # flatten data
            data = array(history)
            data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
            yhat_sequence = list()
            for j in arange(1,NhrBinsPerDay+1):
                # retrieve last observations for input data     
                if j<NhrBinsPerDay:        
                    input_x = data[-self.n_input-NhrBinsPerDay+i:-NhrBinsPerDay+i, :-1]            
                else:
                    input_x = data[-self.n_input:, :-1]            
                # reshape into [1, n_input, 1]
                input_x = input_x.reshape((1, len(input_x), input_x.shape[1]))
                # forecast the next day
                yhat = model.predict(input_x, verbose=0)
                # we only want the vector forecast
                yhat = yhat[0]
                yhat_sequence.append(yhat)                       
            # store the predictions
            predictions.append(yhat_sequence)		            
        predictions = array(predictions)
        predictions = predictions.reshape(predictions.shape[0],predictions.shape[1]) 
        return predictions