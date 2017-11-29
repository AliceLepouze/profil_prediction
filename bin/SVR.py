# -*- coding: utf-8 -*-
"""
@author: alepouze
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import os
import pickle
from sklearn.preprocessing import StandardScaler


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def NRMSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.sqrt(mean_squared_error(y_true, y_pred)))/np.max(y_true) * 100

def Plot_Target(file_name, cluster):
    ''' This function plots the target consumption profil of the wanted 
    cluster.'''
    Target_df = pd.DataFrame(pd.read_pickle('Cluster' + str(cluster) + '_' +
                               file_name + '.pkl'))
    
    Target_np = Target_df.values
    fig = plt.figure()
    plt.plot([k*0.25 for k in range(192)], Target_np[
            len(Target_np)-192:len(Target_np)]) 
    plt.xlabel('Time [h]')
    plt.ylabel('Consumption [Wh]')
    plt.title('Target ' + str(cluster))
#    plt.savefig('Target ' + str(cluster))
    plt.show()

def main(file_name, cluster):
    ''' This function returns the trained model. '''
    
    Target_df = pd.DataFrame(pd.read_pickle('Adapted_Cluster' + str(cluster) +
                                            '_' + file_name + '.pkl'))
    Data_all_df = pd.read_pickle('Features_Cluster' + str(cluster) +
                                   '_' + file_name + '.pkl')

    Data_all_np = Data_all_df.values
    Target_np = Target_df.values
    # ensure all data is float
    Data_all_np = Data_all_np.astype('float64')
    Target_np = Target_np.astype('float64')    
    #Normalize the target #
    scaler = StandardScaler()
    Target_normal_np = scaler.fit_transform(Target_np)
    Target_normal_np = Target_normal_np.ravel()
    
    #Split the data #
    X_train, X_test, y_train, y_test = train_test_split(
            Data_all_np, Target_normal_np, test_size = 0.3, 
            random_state=42, shuffle = False)
    
    # Setup the hyperparameter grid
    param_dist = {"C": [0.001, 0.01], #[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "epsilon": [0.0001, 0.001], #[0.0001, 0.001, 0.01, 0.1, 0.5],
                  "kernel": ['rbf']}

     #Instantiate the RandomizedSearchCV object
    model = SVR()
    model_cv = RandomizedSearchCV(model, param_dist, cv=5, n_iter=4, n_jobs=2)
    model_cv.fit(X_train, y_train)
    
     #Print the tuned parameters and score
    print("Tuned SVR Parameters: {}".format(model_cv.best_params_))
    print("Best score is {}".format(model_cv.best_score_))
    
    model_final = SVR(kernel='rbf', 
                      C = model_cv.best_params_['C'], 
                      epsilon = model_cv.best_params_['epsilon'])

    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)

    # Invert scaling for forecast
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate MAPE   
    print("Mean absolute percentage error: {}".format(
            MAPE(y_test_inv, y_pred_inv)))
    print("Normalized Root Mean Squared Error: {}".format(
            NRMSE(y_test_inv, y_pred_inv)))

    plt.plot([k*0.25 for k in range(96*7)], y_test_inv[
            len(y_test_inv)-96*7:len(y_test_inv)], 'b')
    plt.plot([k*0.25 for k in range(96*7)], y_pred_inv[
            len(y_pred_inv)-96*7:len(y_pred_inv)], 'r')
    plt.show()
    return model_final


if __name__ == '__main__':
    file_name = 'dataset1'
    cluster = 1
    model = main(file_name, cluster)
#    filename = 'SVR_model.pkl'
#    pickle.dump(model, open(filename, 'wb'))
    # load the model from disk
#    loaded_model = pickle.load(open(filename, 'rb'))










