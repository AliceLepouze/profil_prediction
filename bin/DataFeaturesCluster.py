# -*- coding: utf-8 -*-
"""
@author: alepouze
"""

# =============================================================================
# ### The dataset must begin at 00:00 ###
# =============================================================================

import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import itertools

def Weather(weather_file_name):
    ''' This function saves the csv file containing the weather into a pickle 
    file. '''
    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, '../data/' + weather_file_name + '.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    
    with open(filename) as f:
        n_cols = len(f.readline().split(','))
        
    weather_df = pd.read_csv(filename, delimiter=',', 
                             usecols=range(1,n_cols))
    weather_df = weather_df.drop(['pressure', 'humidity', 'irradiance'], 
                                 axis=1)
    weather_df.to_pickle(weather_file_name + '.pkl')


def Feature_Time(file_name): 
    ''' This function prepares the data time features for 
    the neural network.''' 
    
    date_time1_df = pd.read_pickle('Time_' + file_name + '.pkl')    
    FeaturesTime_temp_df = pd.concat([date_time1_df['Month'], 
                                 date_time1_df['Weekday'], 
                                 date_time1_df['Hour']], axis=1)
    OneHot = OneHotEncoder(sparse=False)
    FeaturesTimeOneHot_df = pd.DataFrame(OneHot.fit_transform(
            FeaturesTime_temp_df.values))
    FeaturesTime_df = pd.concat([FeaturesTimeOneHot_df, 
                                 date_time1_df['Holidays']], axis=1)  
    return FeaturesTime_df

def Feature_PreviousReading(file_name, cluster, nb_preceding_time):
    ''' This function returns the features concerning the previous time 
    readings of the previous day and previous week.
        file_name = name of the dataset file
        cluster = integer indicating the cluster on which the features are 
        created. 
        nb_preceding_time = number of preceding time steps.  '''    
    
    Target = pd.read_pickle('Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    Feature_PreviousReading = pd.DataFrame()
    time_step = 24*4
    for preceding in range(nb_preceding_time):
        target = []
        target = [0]*(time_step + preceding) + Target.tolist()[
                0:-(time_step + preceding)]
        Feature_PreviousReading['PreviousDay_' + str(preceding) + 
                                'Preceding' ] = pd.Series(target).values     
    for preceding in range(nb_preceding_time):
        target = []
        target = [0]*(time_step*7 + preceding) + Target.tolist()[
                0:-(time_step*7 + preceding)]
        Feature_PreviousReading['PreviousWeek_' + str(preceding) + 
                                'Preceding' ] = pd.Series(target).values   
    return Feature_PreviousReading
                    
def Feature_MeanConsumption(file_name, cluster):
    ''' This function returns the features concerning the mean consumption 
    of the previous day
        file_name = name of the dataset file
        cluster = integer indicating the cluster on which the features are 
        created. '''        
     
    Target = pd.read_pickle('Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    Feature_MeanConsumption = pd.DataFrame()
    time_step = 24*4
    mean_previous = [Target.iloc[k:k + time_step].mean() for k in range(0, 
                len(Target)-time_step, time_step)]
    mean_day = [0]*(time_step) + np.repeat(
            mean_previous, time_step).tolist()[0:len(Target) - time_step]
    Feature_MeanConsumption['Mean_Previous_Day'] = pd.Series(mean_day).values
    return Feature_MeanConsumption

def Feature_MeanPrevious(file_name, cluster, nb_previous_days, 
                         nb_preceding_time):
    ''' This function returns the features concerning the average consumption 
    of several previous days for each hour .
        file_name = name of the dataset file
        cluster = integer indicating the cluster on which the features are 
        created. 
        nb_previous_days = number of previous days taken into account for 
        the average.
        nb_preceding_time = number of preceding time steps.  '''    

    Target = pd.read_pickle('Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    Feature_MeanPrevious = pd.DataFrame()
    time_step = 24*4
    mean_days_previous = [[Target.iloc[k+i : k+i+time_step*nb_previous_days : 
        time_step].mean() for i in range(96)] for k in range(0, 
            len(Target)-time_step*nb_previous_days, time_step)]
    mean_days_previous = list(itertools.chain.from_iterable(
            mean_days_previous))
    for preceding in range(nb_preceding_time): 
        days_mean=[]
        days_mean = [0]*(
                time_step*nb_previous_days + preceding) + mean_days_previous[
                0:len(Target) - (time_step*nb_previous_days + preceding)]
        Feature_MeanPrevious[
                'Mean_Previous_' + str(nb_previous_days) + 'Days_' + 
                str(preceding) + 'Preceding'] = pd.Series(days_mean).values
    return Feature_MeanPrevious
           
def Feature_LastValues(file_name, cluster, nb_last_values):
    ''' This function returns the features concerning the last values 
    of the previous day.
        file_name = name of the dataset file
        cluster = integer indicating the cluster on which the features are 
        created. '''        
    
    Target = pd.read_pickle('Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    Feature_LastValues = pd.DataFrame()
    time_step = 24*4
    for last_val in range(1,nb_last_values):
        last = []
        last = [Target.iloc[k] for k in range(time_step-last_val, 
                len(Target), time_step)]
        last_value = [0]*(time_step) + np.repeat(last, time_step).tolist()[
                0:len(Target) - time_step]
        Feature_LastValues['Last_Value_' + str(last_val)
        ] = pd.Series(last_value).values
    return Feature_LastValues
                       
    
def main(file_name, weather_file_name, cluster):
    ''' This function creates the normalized features for the given cluster and 
    data set '''
    Target_df = pd.read_pickle('Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    Weather(weather_file_name)
    Weather_df = pd.read_pickle(weather_file_name + '.pkl')
    Feature_Time_df = Feature_Time(file_name)
    Feature_PreviousReading_df = Feature_PreviousReading(file_name, cluster, 
                                                         nb_preceding_time)
    Feature_MeanConsumption_df = Feature_MeanConsumption(file_name, cluster)
    Feature_MeanPrevious_df = Feature_MeanPrevious(file_name, 
                                                   cluster, nb_previous_days, 
                                                   nb_preceding_time)
    Feature_LastValues_df = Feature_LastValues(file_name, cluster, 
                                               nb_last_values) 
    FeaturesFull_df = pd.concat([Weather_df, Feature_PreviousReading_df, 
                                 Feature_MeanConsumption_df, 
                                 Feature_MeanPrevious_df,
                                 Feature_LastValues_df], axis=1)
    # Normalize the features #
    scaler = StandardScaler()
    FeaturesFull_Normalized_df = pd.DataFrame(scaler.fit_transform(
            FeaturesFull_df.values))
    # Add the time features #
    FeaturesFull_Complete_df = pd.concat([FeaturesFull_Normalized_df, 
                                          Feature_Time_df], axis=1)
    # Remove the first two weeks #
    time_step = 24*4
    list_week = [k for k in range(time_step*7*2)]
    FeaturesFull_Complete_df = FeaturesFull_Complete_df.drop(
            FeaturesFull_Complete_df.index[list_week])
    Target_df = Target_df.drop(Target_df.index[list_week])

    FeaturesFull_Complete_df.to_pickle('Features_Cluster' + str(cluster) +
                                   '_' + file_name + '.pkl')
    Target_df.to_pickle('Adapted_Cluster' + str(cluster) + '_' + 
                            file_name + '.pkl')
    
    
if __name__ == '__main__':
    file_name = 'dataset1'
    cluster = 1
    weather_file_name = 'weather'
    nb_preceding_time = 1
    nb_previous_days = 3
    nb_last_values = 1
    main(file_name, weather_file_name, cluster)    


    
            
    
    
    
            
            

