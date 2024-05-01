#!/usr/bin/env python
# coding: utf-8

# In[28]:


import heartpy as hp 
import numpy as np
import pandas as pd
import os
PATH = "/scratch/alim/overnight_validation/MUSE-PSG/"
SAMPLE_TIME_IN_SECONDS = 10
PSG_SAMPLING_RATE = 64
import math
from datetime import datetime
now = datetime.now()
print(f"Started job at {now}")
EVENT_LABELS  = ["Mixed Apnea", "Central Apnea", "Obstructive Apnea",'Arousal (ARO SPONT)', 'RERA', 'Arousal (ARO RES)',\
                 'Limb Movement (Right)' ,'Arousal (ARO Limb)' ,'Limb Movement (Left)'\
                    ,'PLM (Right)', 'PLM (Left)' ,'SpO2 desaturation']


# In[29]:


def read_files_in_directory(directory_path):
    try:
        #print("Finding all MUSE-PSG files")
        # Get a list of all files in the directory
        file_list = sorted([entry.name for entry in os.scandir(directory_path) if
                    entry.is_file() and any(keyword in entry.name for keyword in ['ppg','acc','events'])])


                     
        # Read the contents of each file
        #print("Findind all file names completed")

        return file_list
    except Exception as e:
        print(f"Error reading files: {e}")

def convertList_to_PatienttoDictionary(file_list,dir_path):
    patient_dictionary = {}
    for filename in file_list:
        parts = filename.rsplit('_',1)
        patient_name = parts[0]
    
        if not patient_name in patient_dictionary:
            patient_dictionary[patient_name] = {}
            
        if 'ppg' in filename:
            patient_dictionary[patient_name]['ppg'] = dir_path + filename

        elif 'acc' in filename:
            patient_dictionary[patient_name]['acc'] = dir_path + filename

        elif 'events' in filename:
            patient_dictionary[patient_name]['events'] = dir_path + filename

    return patient_dictionary





# In[30]:


file_name_lst = read_files_in_directory(PATH)
patient_dictionary = convertList_to_PatienttoDictionary(file_name_lst,PATH)



# THE CODE BELOW IS SPECIFIC FOR EXTRACTING FEATURES FROM PPG SIGNALS

# In[32]:


def cleanedPpgSignal(signal,cutoff = [0.7,3.5],sample_rate = PSG_SAMPLING_RATE,order = 3,filter_type = 'bandpass'):
    #returns a list representing cleaned signal

    return hp.filter_signal(signal,cutoff = cutoff,sample_rate = sample_rate, order = order, filtertype = filter_type)
        

def processPpgSignal(signal,sampling_rate = PSG_SAMPLING_RATE):

    wd,m = hp.process(signal,sample_rate = sampling_rate)
    return wd,m


def getMeanDistanceBetweenPeaksinSeconds(peaks_indices,sampling_rate = PSG_SAMPLING_RATE):
    #note: you can use this on the negated np array and get the mean distnace between troughs
    
    if len(peaks_indices) <= 1:
        return 0

    sum_distances = 0
    for i in range(len(peaks_indices)-1):

        current = peaks_indices[i]
        next = peaks_indices[i+1]
        distance = next - current

        sum_distances += distance

    return (sum_distances/(len(peaks_indices)-1))/sampling_rate
    

    
def getAverageDistanceBetweenPeakandNextTroughinSeconds(peaks_indices,trough_indices,sampling_rate = PSG_SAMPLING_RATE):
    #this is for time distance
    if (len(peaks_indices)== 0) or (len(trough_indices)==0):
        return None
    iterations = len(peaks_indices)
    iterations = min(iterations,len(trough_indices))
    sum_distances = 0

    if not (peaks_indices[0] < trough_indices[0]):
        if len(trough_indices)<=1:
            return 0
        trough_indices = trough_indices[1:]
        iterations = min(iterations,len(trough_indices))



    for i in range(iterations):
        sum_distances += abs(peaks_indices[i] - trough_indices[i])
    return (sum_distances/sampling_rate)/iterations

def isUsable(df,exluded_columns):
    #excluded_columns is a list of strings
    for index, row in df.iterrows():
        for col_name, value in row.items():
            if not col_name in exluded_columns:
                if (value == np.nan) or (value == None) or (value == "masked"):
                    return False
            
    return True



# In[33]:


import warnings
warnings.filterwarnings("ignore")

acceptable_length_of_signal_in_seconds = 9.5
column_names = ['patient_Id','mean_time_between_peaks (S)',\
                'mean_time_between_troughs(S)','mean_time_between_peak_and_next_trough (S)',\
                    'mean_time_between_trough_and_next_peak (S)','bpm', 'ibi', 'sdnn', 'sdsd', \
                        'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate','std','mean','var','overlapping_events']
return_Df = pd.DataFrame(columns=column_names)

for patientName in patient_dictionary:
    #load data from .csv
    try:
        ppg_fileName = patient_dictionary[patientName]['ppg']
        events_fileName = patient_dictionary[patientName]['events']
        ppg_df = pd.read_csv(ppg_fileName)
        events_df = pd.read_csv(events_fileName)

    except Exception as e:
        print(f"Failed to load data")
        continue
    #normalize the ppg_df['ch2']
    ppg_df.loc[:, 'ch2'] = (ppg_df['ch2'] - ppg_df['ch2'].min(axis=0)) / (ppg_df['ch2'].max(axis=0) - ppg_df['ch2'].min(axis=0))

    #add datetime format
    ppg_df['ts-datetime'] = pd.to_datetime(ppg_df['ts'],unit = 's')
    events_df['start-datetime'] = pd.to_datetime(events_df['start'],unit = 's')
    events_df['end-datetime'] = pd.to_timedelta(events_df['duration'],unit = 's') + events_df['start-datetime']

    #initiate accumulates for apnea and non-apnea
    event_matrix = pd.DataFrame(columns = column_names)
    non_event_matrix = pd.DataFrame(columns = column_names)

    #find desired event labels
    events = events_df

    #set 'ts-datetime' to index (needed for later splitting into intervals)
    ppg_df.set_index('ts-datetime', inplace=True)

    #this is the length of the interval that the signal will be split into 
    freq = f"{SAMPLE_TIME_IN_SECONDS}S"

    # Group the DataFrame by the specified frequency
    ppg_10s_intervals = ppg_df.groupby(pd.Grouper(freq=freq))

    #iterate through every interval in ppg_df
    for interval, data in ppg_10s_intervals:
        try:
            overlapping_events = events[(events['start-datetime'] <= interval  + pd.to_timedelta(SAMPLE_TIME_IN_SECONDS,unit = 's')) & (events['end-datetime'] >=interval)]
            ppg_signal = data['ch2']
            if(len(ppg_signal) < acceptable_length_of_signal_in_seconds * PSG_SAMPLING_RATE):
                continue
            isEvent = not overlapping_events.empty
            
            label = overlapping_events['name'].unique()
            #attempt to clean ppg_signal
            cleaned_ppg_signal = cleanedPpgSignal(ppg_signal)


            #get negative cleaned_ppg_signal
            negative_cleaned_ppg_signal = -1 * cleaned_ppg_signal

            #attempt to process ppg_signal
            processed = processPpgSignal(cleaned_ppg_signal)

            #attempt to process negative ppg_signal
            negative_processed = processPpgSignal(negative_cleaned_ppg_signal)

            #extract dictionaries out of tuples
            wd,m = processed[0],processed[1]
            neg_wd, neg_m = negative_processed[0],negative_processed[1]

            #declare peak list and trough list
            peak_list = wd['peaklist']
            trough_list = neg_wd['peaklist']

            #get mean distance between peaks
            mean_distance_between_peaks = getMeanDistanceBetweenPeaksinSeconds(peaks_indices = peak_list)


            #get mean distance between troughs
            mean_distance_between_troughs = getMeanDistanceBetweenPeaksinSeconds(peaks_indices=trough_list)

            #get mean difference between peak trough after peak
            mean_difference_between_peak_and_next_trough_in_seconds = getAverageDistanceBetweenPeakandNextTroughinSeconds(peaks_indices=peak_list,trough_indices=trough_list)


            #get mean difference between trough and peak after trough
            mean_difference_between_trough_and_next_peak_in_seconds = getAverageDistanceBetweenPeakandNextTroughinSeconds(peaks_indices=trough_list,trough_indices=peak_list)



            #extract std, mean, and var
            std = np.std(cleaned_ppg_signal)
            mean = np.mean(cleaned_ppg_signal)
            var = np.var(cleaned_ppg_signal)

            feature_row = {
                    'patient_Id':patientName,
                    'std':std,
                    'mean':mean,
                    'var':var,
                    'overlapping_events':label,
                    'mean_time_between_peaks (S)':mean_distance_between_peaks,
                    'mean_time_between_troughs(S)':mean_distance_between_troughs,
                    'mean_time_between_peak_and_next_trough (S)':mean_difference_between_peak_and_next_trough_in_seconds,
                    'mean_time_between_trough_and_next_peak (S)':mean_difference_between_trough_and_next_peak_in_seconds

                }
            
            #join feature_row and m
            feature_row = {**m,**feature_row}
            feature_row = pd.DataFrame([feature_row])
            feature_row.head()

            if isUsable(feature_row,['overlapping_events']):
                    #print(f"Successfully processed!")
                if isEvent:
                    event_matrix = pd.concat([event_matrix,feature_row])
            
                else:
                    non_event_matrix = pd.concat([non_event_matrix,feature_row])
        except hp.exceptions.BadSignalWarning as e:
            pass

    #add it all together...
    return_Df = pd.concat([return_Df,event_matrix])
    non_event_matrix = non_event_matrix.sample(min(len(event_matrix),len(non_event_matrix)))
    return_Df = pd.concat([return_Df,non_event_matrix])
    print(f"apnea_matrix length: {event_matrix.shape[0]}, non_apnea_matrix length = {non_event_matrix.shape[0]}, matrix length = {return_Df.shape[0]}")

end = datetime.now()
print(f"Finished job at {end}")
end_string = now.strftime('%Y-%m-%d-%H-%M-%S')
return_Df.to_csv(f"allEvents_{end_string}_features.csv",index = False)




        

        

        

        



    
    


# In[18]:




