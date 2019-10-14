# -*- coding: utf-8 -*-
"""
Created on 29 Sep 2019

# Example script that trains and tests an LSTM model to estimate water column using handpump vibration data features.

# cd to ...\handpumpWaterColEst\scripts folder 
# define pathData = ...\handpumpWaterColEst\data
# define pathSave = ...\handpumpWaterColEst\outputs

@author: Achut Manandhar
Adapted from the following example:
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

"""
#cd ...\handpumpWaterColEst\scripts
from LSTMutils import *
from LSTMmultiInUniOut import *

import os
from numpy import percentile
from pandas import read_csv
from pandas import to_datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model
#from keras.models import load_model

# Plot training and validation loss
def plot_train_valid_loss(modelHistory,pathSave,savePlot=False):
    fig1, ax1 = plt.subplots()
    ax1.plot(modelHistory.history['loss'])
    ax1.plot(modelHistory.history['val_loss'])
    ax1.legend(['Train','Validation'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs',fontsize='x-large')
    ax1.set_ylabel('RMSE',fontsize='x-large')    
    ax1.set_title('Train vs Valid Loss',fontsize='x-large') 
    if savePlot:
        fig1.savefig(os.path.join(pathSave,'lossTrainValid.png'),dpi=300)

# Plot LSTM training, validation, and test estimates
def plot_lstm_outputs(lstm_obj,model,train,test,zmuvY,pathSave,savePlot=False):  
    # Pred on train
    pred_train = lstm_obj.predict_model(model, train, train, NhrBinsPerDay) 
    # Pred on test
    pred = lstm_obj.predict_model(model, train, test, NhrBinsPerDay)     
    
    print(train.shape)
    print(pred_train.shape)
    print(test.shape)
    print(pred.shape)               
        
    pred_train_zinv = zmuvY.inverse_transform(pred_train)
    pred_zinv = zmuvY.inverse_transform(pred)
    print(pred_train_zinv.shape)
    print(pred_zinv.shape)
    
    # Train/Valid/Test
    NdaysTrain = pred_train.shape[0]
    NtrainValid = int((1-lstm_obj.validation_split)*NdaysTrain*NhrBinsPerDay)
    NtrainTest = int(NdaysTrain*NhrBinsPerDay)
    fig2, ax2 = plt.subplots()
    ax2.plot(dataset.index[:NtrainValid].values,pred_train_zinv[:NtrainValid],'.')
    ax2.plot(dataset.index[NtrainValid:NtrainTest].values,pred_train_zinv[NtrainValid:],'.')
    ax2.plot(dataset.index[NtrainTest:].values,pred_zinv,'.')
    ax2.plot(dataset['Ybin'])
    ax2.legend(['Estimate-Train','Estimate-Valid','Estimate-Test','Truth'])
    ax2.set_ylabel('Water Column (m)')
    ax2.set_title('(Tsteps={},Nh={},Nb={},Ep={},Lr={})'.format(lstm_obj.n_input,
                  lstm_obj.n_hid,
                  lstm_obj.batch_size,
                  lstm_obj.n_epoch,
                  lstm_obj.learn_rate),fontsize='x-large') 
    if savePlot:
        fig2.savefig(os.path.join(pathSave,'waterColEstimates.png'),dpi=300)             
    
# Pre-process data
def preprocess_data(dataset):
    # Bin water column labels
    dataset['Ybin'] = waterColBinFun(dataset['waterColY'].values)
    Ytruth = dataset.pop('waterColY') 
    
    # Plot daily fetures and labels
    plt.imshow(dataset.values.T,aspect='auto')
    clims = percentile(dataset.values.reshape(dataset.shape[0]*dataset.shape[1],),[2.5,97.5])
    plt.clim(vmin=clims[0], vmax=clims[1])
        
    NhrBinsPerDay = 1 # Assuming daily average over hour bins per day
    
    # Split data into train/test
    NdaysTrain = int(dataset.shape[0]*2/3)
    train, test = split_dataset(dataset.values,NdaysTrain,NhrBinsPerDay)
    print(train.shape)
    print(test.shape)
    
    # Standardize dataset
    train, test, zmuv, zmuvY = standardize_dataset(train,test,NhrBinsPerDay)
    
    # verify train data
    print(train.shape)
    print(train[0, 0, 0], train[-1, -1, 0])
    # verify test
    print(test.shape)
    print(test[0, 0, 0], test[-1, -1, 0])
    
    return train, test, zmuv, zmuvY, NhrBinsPerDay
    
# Test    
def evaluate_model(train,test,NhrBinsPerDay,pathSave,savePlot=False):                  
    # Prepare data in LSTM format
    n_timesteps = 3
    train_x, train_y = to_supervised(train, n_timesteps)
    test_x, test_y = to_supervised(test, n_timesteps)
    
    lstm_obj = LSTMmultiInUniOut(n_input = n_timesteps, 
                              n_hid = 50, 
                              dropout = 0.2, 
                              validation_split = .2, 
                              loss_fn = 'mse', 
                              optimizer = Adam, 
                              learn_rate = .0001,
                              L1L2 = [0.0,0.0001], 
                              batch_size = 10, 
                              n_epoch = 50, 
                              verbose = 1, 
                              shuffle = 0)
    
    model = lstm_obj.build_model(train_x, train_y) 
    
    # summarize layers
    print(model.summary())
    # plot graph
    if savePlot:
        plot_model(model, to_file=os.path.join(pathSave,'model_graph.png'))
    
    modelHistory, model = lstm_obj.train_model(model, train_x, train_y)
    pred = lstm_obj.predict_model(model, train, test, NhrBinsPerDay)     
    scores = score_model(test_y,pred)
    
    return scores, pred, modelHistory, model, lstm_obj


###############################################################################
# Read dataset
pathData = r'...\handpumpWaterColEst\data'
pathSave = r'...\handpumpWaterColEst\outputs'    
fileName = 'dailyWaterColFreqFeatsPumpMP1.csv'
        
dataset = read_csv(os.path.join(pathData,fileName))
dataset['date'] = to_datetime(dataset['date'],dayfirst=True)
dataset = dataset.set_index('date')

# Downsample frequency bins   
dataset = dataset.drop(['avgCwtX2', 'avgCwtX4', 'avgCwtX6', 'avgCwtX8', 'avgCwtX10',
'avgCwtX12', 'avgCwtX14', 'avgCwtX16', 'avgCwtX18', 'avgCwtX20'],axis=1) 

###############################################################################    
# Preprocess - Bin, split, standardize    
train, test, zmuv, zmuvY, NhrBinsPerDay = preprocess_data(dataset)    
    
###############################################################################
# Build, train, test model
scores, pred, modelHistory, model, lstm_obj = evaluate_model(train,test,NhrBinsPerDay,pathSave,savePlot=False)

# Uncomment to save
# save the entire model (archi, weights,...) as a HDF5 file
#model_file = 'modelH100E50B10L0_0001.h5'
#filenameModel = os.path.join(pathSave,model_file)
#model.save(filenameModel)
## load saved model
#loaded_model = load_model(os.path.join(pathSave,model_file))

###############################################################################
# Plot outputs
# Plot training and validation loss
# Change savePlot=True to save the plot
plot_train_valid_loss(modelHistory,pathSave,savePlot=False)

# Plot water column estimates
# Change savePlot=True to save the plot
plot_lstm_outputs(lstm_obj,model,train,test,zmuvY,pathSave,savePlot=False)

