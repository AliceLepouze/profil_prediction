# -*- coding: utf-8 -*-
"""
@author: alepouze
"""
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from SVR import MAPE, NRMSE, Plot_Target
from sklearn.preprocessing import StandardScaler
import pickle
from keras.models import load_model


def mainCluster(file_name, cluster):
    ''' This function returns the trained model predicting a cluster
    consumption. '''
    
    Target_df = pd.DataFrame(pd.read_pickle('Adapted_Cluster' + str(cluster) +
                                            '_' + file_name + '.pkl'))
    Data_all_df = pd.read_pickle('Features_Cluster' + str(cluster) +
                                   '_' + file_name + '.pkl')

    Data_all_np = Data_all_df.values
    Target_np = Target_df.values
    # Ensure all data is float
    Data_all_np = Data_all_np.astype('float64')
    Target_np = Target_np.astype('float64')    
    # Normalize the target #
    scaler = StandardScaler()
    Target_normal_np = scaler.fit_transform(Target_np)
    Target_normal_np = Target_normal_np.ravel()

    # Spilt the data #
    n_train_hours = 365*2 * 24*4
    n_val_hours = 365 * 24*4
    X_train = Data_all_np[:n_train_hours, :]
    y_train = Target_normal_np[:n_train_hours]
    X_val = Data_all_np[n_train_hours: n_train_hours + n_val_hours, :]
    y_val = Target_normal_np[n_train_hours: n_train_hours + n_val_hours]
    X_test = Data_all_np[n_train_hours + n_val_hours:, :]
    y_test = Target_normal_np[n_train_hours + n_val_hours:]
    
    # Save the number of columns in predictors: n_cols
    n_cols = X_train.shape[1]
    # Set up the model: model
    inputs = Input(shape=(n_cols,))
    h1 = Dense(60, activation='sigmoid')(inputs)
    h2 = Dense(60, activation='sigmoid')(h1)
    outputs =  Dense(1, activation='linear')(h2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam',
              loss='mean_absolute_percentage_error')
#    print(model.summary())

    # Fit the model 
    history = model.fit(X_train, y_train, epochs=50, shuffle = False,
                        validation_data=(X_val, y_val), verbose=False)
    y_pred = model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(y_pred.ravel())
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate MAPE   
    print("Mean absolute percentage error: {}".format(
            MAPE(y_test_inv, y_pred_inv)))
#    print("Normalized Root Mean Squared Error: {}".format(
#            NRMSE(y_test_inv, y_pred_inv)))

    plt.plot([k*0.25 for k in range(96*7)], y_test_inv[
            len(y_test_inv)-96*7:len(y_test_inv)], 'b')
    plt.plot([k*0.25 for k in range(96*7)], y_pred_inv[
            len(y_pred_inv)-96*7:len(y_pred_inv)], 'r')
    plt.show()
    return model

def SumCluster(file_name, list_conso):


def mainConsumer(file_name):
    ''' This function returns the trained model predicting a consumer
    consumption. '''
    
    Data_all_df = pd.read_pickle('Features_' + file_name + '.pkl')
    
    
    

if __name__ == '__main__':
    start_time = time.clock()
    file_name = 'dataset1'
    cluster = 1
    model = main(file_name, cluster)
#    filename = 'MLP_model.h5'
#    model.save(filename)
    print(time.clock() - start_time, "seconds")


    # Load model 
#    my_model = load_model(filename)
#    predictions = my_model.predict(data_to_predict_with)



















