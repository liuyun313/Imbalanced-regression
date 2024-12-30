# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:03:44 2024

@author: 58211
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import regularizers
import sys

def sera(y, y_hat, sample_weight):
    
    min_w = min(sample_weight)
    max_w = max(sample_weight)
    
    t = np.linspace(max_w, min_w, 21)
    
    sera_ = 0
    
    y = np.array(y)
    y_hat = np.array(y_hat)

    
    for i in t:
        index = np.where(sample_weight > i)

        if np.size(index) > 0:

            sera_ = sera_ + mean_squared_error(y[index[0]], y_hat[index[0]])
            
    return sera_

def evaluation(y_test, y_hat, sample_weights_test, y_val, y_hat_val, sample_weights_val):
    
    sera_test = sera(y_test, y_hat, sample_weights_test)        
    
    sera_val = sera(y_val, y_hat_val, sample_weights_val)        
    
    sera_test = np.around(sera_test, decimals=6)
    sera_val = np.around(sera_val, decimals=6)
    
    #result = {}
    #result['sera_test'] = sera_test
    #result['sera_val'] = sera_val
    #result['mse_test'] = mse_test
    #result['mse_val'] = mse_val
    
    return sera_val, sera_test

def regression(X_train, X_val, y_train, y_val, X_test, y_test, importance_type, rg):
    
    best_sera_val = sys.float_info.max
    best_prediction = y_test
    
    factors = [0.0 + i * 0.2 for i in range(20)]
    
    result = []
    result.append(y_test)
    
    result_val = {}
    
    #y_train_norm = np.array((y_train - min(y_train)) / (max(y_train) - min(y_train)))

    for factor in factors:
        
        # Construct importance function
        if importance_type == 'upper':
            sample_weights = np.exp(factor * y_train)
            sample_weights = sample_weights / np.mean(sample_weights)
            sample_weights_test = y_test
            sample_weights_val = y_val

        elif importance_type == 'lower':
            sample_weights = np.exp(factor*(1 - y_train))
            sample_weights = sample_weights / np.mean(sample_weights)
            sample_weights_test = 1 - y_test
            sample_weights_val = 1- y_val
            
        elif importance_type == 'both':
            sample_weights = [np.exp(factor * i) if i >=0.5 else np.exp(factor*(1 - i)) for i in y_train]
            sample_weights = sample_weights / np.mean(sample_weights)
            sample_weights_test = [i if i >=0.5 else 1 - i for i in y_test]
            sample_weights_val = [i if i >=0.5 else 1 - i for i in y_val]

        # Construct regression model
        if rg == 'lr':
            regr = LinearRegression()
            regr.fit(X_train, y_train, sample_weight = sample_weights)
            
        elif rg == 'svr':
            regr = SVR(C=1.0, epsilon=0.1)
            regr.fit(X_train, y_train, sample_weight = sample_weights)
            
        elif rg == 'rf':
            regr = RandomForestRegressor(random_state=0)
            regr.fit(X_train, y_train, sample_weight = sample_weights)
            
        elif rg == 'gbr':
            regr = GradientBoostingRegressor(random_state=0)
            regr.fit(X_train, y_train, sample_weight = sample_weights)
            
        elif rg == 'nn':
            regr = Sequential()     
            regr.add(Dense(units=16, activation='relu', input_shape=(np.size(X_train,1),), kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),))
        
            regr.add(Dense(units=8, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),))
                              
            regr.add(Dense(units=1,  activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),))
            
            lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2,
                decay_steps=100,
                decay_rate=0.9)  
            
            
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=200, 
                verbose=0, 
                mode='auto',
                restore_best_weights = True
            )
    
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)    
            
            regr.compile(optimizer=optimizer, loss = tf.keras.losses.MeanSquaredError())
            regr.fit(X_train, y_train, 
                      batch_size=32, 
                      epochs=2000, 
                      callbacks=[early_stopping],
                      validation_data=(X_val, y_val),
                      sample_weight = sample_weights)
        
        
        y_hat = regr.predict(X_test)        
        y_hat_val = regr.predict(X_val)
        
        sera_val, sera_test = evaluation(y_test, y_hat, sample_weights_test, y_val, y_hat_val, sample_weights_val)
        result_val[str(factor)] = sera_val
        
        if factor == 0.0:
            result.append(y_hat)
            result.append(sera_test)
            result.append(np.around(mean_squared_error(y_hat, y_test), decimals=6))
            
        else:                    
            if sera_val < best_sera_val:
                
                best_sera_val = sera_val
                best_prediction = y_hat        
    
    
    return best_prediction


def importance_weighting(X_train, X_test, y_train, y_test, importance_type, rg):



    '''
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        test_size=0.2, 
                                                        random_state=0)  #划分训练集和测试集
    
    best_prediction = regression(X_train, X_val, y_train, y_val, X_test, y_test, importance_type, rg)
    
    return best_prediction



