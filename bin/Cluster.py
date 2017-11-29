# -*- coding: utf-8 -*-
"""
@author: alepouze
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import datetime as dt
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def SaveData(file_name):
    ''' This function saves the csv file containing the data into a pickle 
    file.
    From the csv file, the function removes the first column containing the 
    date and the first row containing the headers. Then, it adds to each 
    column a consumer name from Consumer0 to ConsumerN, N being the number of 
    consumers in the file.'''
    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, '../data/' + file_name + '.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    
    with open(filename) as f:
        ncols = len(f.readline().split(','))
        
    df_consumer = pd.read_csv(filename, delimiter=',', skiprows=1, 
                     usecols=range(1,ncols), 
                     names = ["Consumer" + str(i-1) for i in range(ncols)])
    
    df_consumer.to_pickle(file_name + '.pkl')


def SaveDataDate(file_name):
    ''' This function saves the dates and time linked to each consumer 
    consumption. '''
    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, '../data/'  + file_name + '.csv')
    filename = os.path.abspath(os.path.realpath(filename))
        
    df_date = pd.read_csv(filename, delimiter=',', skiprows=1, 
                          usecols=[0], names =['Date'])

    df_date['Date'] = pd.to_datetime(df_date['Date'], 
           format="%Y-%m-%dT%H:%M:%SZ")
    df_date['DateOnly'] = pd.DatetimeIndex(df_date['Date']).date
    df_date['Year'] = pd.DatetimeIndex(df_date['Date']).year
    df_date['Month'] = pd.DatetimeIndex(df_date['Date']).month
    df_date['Day'] = pd.DatetimeIndex(df_date['Date']).day
    df_date['Hour'] = pd.DatetimeIndex(df_date['Date']).hour
    df_date['Minute'] = pd.DatetimeIndex(df_date['Date']).minute
    df_date['DayYear'] = [int(format(i, '%j')) for i in df_date['Date']] 
    df_date['Weekday'] = df_date['Date'].dt.dayofweek 
    # Monday is 0 in "Weekday"
    df_date['BusinessDay'] = pd.Series([int(x>4) for x in df_date['Weekday']])
    df_date['MinuteInHour'] = df_date['Minute']/60 # Minutes express in hour
    df_date['TimeHour'] = df_date['Hour'] + df_date['MinuteInHour']
    # Write manually the holidays in the city where the dataset was taken
    holidays = []
    for i in [2014,2015,2016,2017]:
        holidays.extend([dt.date(i,1,1), dt.date(i,5,1), dt.date(i,8,1), 
                         dt.date(i,12,25), dt.date(i,12,26)]) 
    holidays.extend([dt.date(2014,4,18), dt.date(2014,4,21), 
                     dt.date(2014,5,29), dt.date(2014,6,9), dt.date(2015,4,3), 
                dt.date(2015,4,6), dt.date(2015,5,14), dt.date(2015,5,25),
                dt.date(2016,3,25), dt.date(2016,3,28), dt.date(2016,5,5), 
                dt.date(2016,5,16), dt.date(2017,4,14), dt.date(2017,4,17), 
                dt.date(2017,5,25), dt.date(2017,6,5)])
    is_holiday = []
    for x in df_date['DateOnly']:
        for i in holidays:
            if x == i:
                is_holiday.append(int(x==i))
                break
            else:
                if i == holidays[-1]:
                    is_holiday.append(0)
    is_holiday_series = pd.Series(is_holiday)
    df_date['Holidays'] = is_holiday_series.values       
    
    df_date.to_pickle('Time_' + file_name + '.pkl')


def SolsticeEquinox(df, year_begin, year_end):
    ''' This function defines solstices and equinoxes from year_begin to 
    year_end of the dataframe df containg the dates and time.
    It started at the winter solstice (December) of year_begin and ends at the
    the winter solstice (December) of year_end'''

    idx_solstice_winter1 = df.index[
            df['DateOnly'] == dt.date(year_begin,12,21)].tolist()[0]
    idx_equinox_spring1 = df.index[
            df['DateOnly'] == dt.date(year_begin+1,3,20)].tolist()[0]
    idx_solstice_summer1 = df.index[
            df['DateOnly'] == dt.date(year_begin+1,6,21)].tolist()[0]
    idx_equinox_autumn1 = df.index[
            df['DateOnly'] == dt.date(year_begin+1,9,22)].tolist()[0]  
    idx_solstice_winter2 = df.index[
            df['DateOnly'] == dt.date(year_begin+1,12,21)].tolist()[0]
    idx_equinox_spring2 = df.index[
            df['DateOnly'] == dt.date(year_end,3,20)].tolist()[0]
    idx_solstice_summer2 = df.index[
            df['DateOnly'] == dt.date(year_end,6,21)].tolist()[0]
    idx_equinox_autumn2 = df.index[
            df['DateOnly'] == dt.date(year_begin+1,9,22)].tolist()[0]  
    
    return [idx_solstice_winter1, idx_solstice_winter2, 
           idx_solstice_summer1, idx_solstice_summer2, idx_equinox_autumn1, 
           idx_equinox_autumn2, idx_equinox_spring1, idx_equinox_spring2]


def ClusterWinterFeatures(file_name):  
    '''  This function builds the cluster features for the winter period 
    of the given file'''

    ### Get the data and the dates and time ###
    data1_df = pd.read_pickle(file_name + '.pkl')
    date_time1_df = pd.read_pickle('Time_' + file_name + '.pkl')
    [idx_solstice_winter1, idx_solstice_winter2, idx_solstice_summer1, 
     idx_solstice_summer2, idx_equinox_autumn1, idx_equinox_autumn2, 
     idx_equinox_spring1, idx_equinox_spring2] = SolsticeEquinox(
     date_time1_df, 2014, 2016)
     
    TimeHour_df = pd.DataFrame(date_time1_df.Hour)
    Weekday_df = pd.DataFrame(date_time1_df.Weekday)
    idx_solstice_winter = [i for i in range(idx_solstice_winter1, 
                                            idx_equinox_spring1)]
    idx_solstice_winter.extend([i for i in range(idx_solstice_winter2, 
                                                 idx_equinox_spring2)])
    TimeHour_df = TimeHour_df.take(idx_solstice_winter)
    TimeHour_df = TimeHour_df.reset_index(drop=True)
    Weekday_df = Weekday_df.take(idx_solstice_winter)
    Weekday_df = Weekday_df.reset_index(drop=True)
    consumer_df = pd.DataFrame(data1_df)
    consumer_df = consumer_df.take(idx_solstice_winter)
    consumer_df = consumer_df.reset_index(drop=True)
    
    FeatureCluster_df = pd.DataFrame()
    for week in [3,5]:
        for j in range(24):
            Mean_Hour = []
            for i in range(data1_df.shape[1]):
                consumer = consumer_df['Consumer' + str(i)].values
                idx = TimeHour_df.index[
                        (TimeHour_df['Hour'] == float(j)) & 
                        (Weekday_df['Weekday'] == week)].tolist()
                consumption_Hour = consumer[idx]
                Mean_Hour.append(mean(consumption_Hour))
            FeatureCluster_df['Hour' + str(j)
            + 'Day' + str(week)] = pd.Series(Mean_Hour).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    FeatureClusterNormal = scaler.fit_transform(FeatureCluster_df.values)
    FeatureClusterNormal = pd.DataFrame(FeatureClusterNormal)

    FeatureClusterNormal.to_pickle('Winter_cluster_Features_' + 
                                   file_name + '.pkl')


def ClusterCategory(nb_of_cluster, file_name):
    ''' This function clusters the data from the file into nb_of_cluster 
    clusters for the wanted season. The resulting dataframe indicates to which
    cluster each consumer is. '''

    Features_df = pd.read_pickle(
            'Winter_cluster_Features_' + file_name + '.pkl')
    model = KMeans(n_clusters=nb_of_cluster).fit(Features_df.values)
    
    ClusterCategory_df = pd.DataFrame()
    ClusterCategory_df['Data_index'] = Features_df.index.values
    ClusterCategory_df['Cluster'] = model.labels_
    ClusterCategory_df.to_pickle('ClusterCategory_' + 
                                   file_name + '.pkl')
    

def main():
    ''' This function creates a pickle file for each cluster. The resulting 
    file is the sum of each consumer consumption within one cluster. '''
    data1_df = pd.read_pickle(file_name + '.pkl')
    ClusterCategory_df = pd.read_pickle('ClusterCategory_' + 
                                   file_name + '.pkl')
    
    for i in range(nb_of_cluster):
        IndexCluster = ClusterCategory_df[
                ClusterCategory_df.Cluster == i]['Data_index'].values.tolist()
        SumCluster = data1_df.T.iloc[IndexCluster]
        Target = SumCluster.T.sum(axis=1)
        Target.to_pickle('Cluster' + str(i) + '_' + 
                         file_name + '.pkl')

if __name__ == '__main__':
    file_name = 'dataset1'
    nb_of_cluster = 4
    SaveData(file_name)
    SaveDataDate(file_name)
    ClusterWinterFeatures(file_name)
    ClusterCategory(nb_of_cluster, file_name)
    main()    


def Inertia(file_name):
    ''' Plot the ideal number of cluster according to the given cluster 
    features. '''
    Features_Cluster = pd.read_pickle(
            'Winter_cluster_Features_' + file_name + '.pkl')
    inertias = []
    ks = range(1, 20)
    for k in ks:
        modelKmeans = KMeans(n_clusters=k).fit(Features_Cluster.values)
        inertias.append(modelKmeans.inertia_)
    
    fig = plt.figure()
    plt.plot(ks, inertias, '-o')
#    plt.plot(4, inertias[3], 'ro')
    plt.xlabel('Number of clusters')
    plt.xticks(ks)
    plt.ylabel('Inertia')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.show()

def Consumers_in_Cluster(file_name, nb_of_cluster):
    ''' This function plots the consumption profil for each consumer belonging
    to the same cluster. '''
    
    data1_df = pd.read_pickle(file_name + '.pkl')
    ClusterCategory_df = pd.read_pickle('ClusterCategory_' + 
                                        file_name + '.pkl')

    for j in range(nb_of_cluster):
        IndexCluster = ClusterCategory_df[
                ClusterCategory_df.Cluster == j]['Data_index'].values.tolist()
        SumCluster = data1_df.T.iloc[IndexCluster]
        fig = plt.figure()
        for i in IndexCluster:
            plt.plot([k*0.25 for k in range(192)], SumCluster.T[
                    'Consumer' + str(i)].iloc[3456-192:3456]) 
            plt.xlabel('Time [h]')
            plt.ylabel('Consumption [Wh]')
        plt.title('Clustering ' + str(j))
    #    plt.savefig(' Cluster ' + str(j))
        plt.show()
        
def ClusterCenter(file_name, nb_of_cluster):
    ''' This function plots the center of each cluster as well as the 
    consumption profil for each consumer belonging to the same cluster. '''
    
    data1_df = pd.read_pickle(file_name + '.pkl')
    ClusterCategory_df = pd.read_pickle('ClusterCategory_' + 
                                   file_name + '.pkl')
    Features= pd.DataFrame()
    for j in range(nb_of_cluster):
        IndexCluster = ClusterCategory_df[
                ClusterCategory_df.Cluster == j]['Data_index'].values.tolist()
        SumCluster = data1_df.T.iloc[IndexCluster]
        Features['Max'] = SumCluster.T.max(axis=1)
        Features['Min'] = SumCluster.T.min(axis=1)
        Features['Centre'] = 0.5*(Features['Max'] + Features['Min'])
        fig = plt.figure()
        for i in IndexCluster:
            plt.plot([k*0.25 for k in range(192)], 
                      SumCluster.T['Consumer' + str(i)].iloc[3456-192:3456], 
                      'b') 
            plt.xlabel('Time [h]')
            plt.ylabel('Consumption [Wh]')
        plt.plot([k*0.25 for k in range(192)], 
                  Features['Centre'].iloc[3456-192:3456], 'r') 
        plt.title('Cluster ' + str(j))
    #    plt.savefig('Cluster_centre ' + str(j))
        plt.show()

def ClusterProfil(file_name, nb_of_cluster):
    ''' This function plots each cluster profil (sum of all consumer 
    consumptions within a cluster). '''
    
    data1_df = pd.read_pickle(file_name + '.pkl')
    ClusterCategory_df = pd.read_pickle('ClusterCategory_' + 
                                   file_name + '.pkl')

    for i in range(nb_of_cluster):
        IndexCluster = ClusterCategory_df[
                ClusterCategory_df.Cluster == i]['Data_index'].values.tolist()
        SumCluster = data1_df.T.iloc[IndexCluster]
        Features = SumCluster.T.sum(axis=1)
        fig = plt.figure()
        plt.plot([k*0.25 for k in range(192)], 
                  Features.iloc[3456-192:3456].values) 
        plt.xlabel('Time [h]')
        plt.ylabel('Consumption [Wh]')
        plt.title('Cluster ' + str(i))
    #    plt.savefig('Cluster_Profile ' + str(i))
        plt.show()
    









