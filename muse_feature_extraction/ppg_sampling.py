#import libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import neurokit2 as nk
import random
path = "/scratch/alim/overnight_validation/MUSE-PSG/"
import os
import random
import pickle


MUSE_SAMPLING_RATE = 64
apnea_labels = ["Mixed Apnea", "Central Apnea", "Obstructive Apnea"]

'''
THE FOLLOWING LENGTH OF CODE LINES 17-83 PROCESSES FILES INTO A USEABLE PATIENT_DATA_DIRECTORY
'''
def read_files_in_directory(directory_path):
    try:
        #print("Finding all MUSE-PSG files")
        # Get a list of all files in the directory
        file_list = sorted([entry.name for entry in os.scandir(directory_path) if
                    entry.is_file() and any(keyword in entry.name for keyword in ['ppg','events'])])


                     
        # Read the contents of each file
        #print("Findind all file names completed")

        return file_list
    except Exception as e:
        print(f"Error reading files: {e}")

# Example usage:
directory_path = "/scratch/alim/overnight_validation/MUSE-PSG"
file_name_list = read_files_in_directory(directory_path)

patient_dictionary = {}

#print("Organizing into dictionary")

for filename in file_name_list:
    parts = filename.rsplit('_',1)
    patient_name = parts[0]
    
    if patient_name in patient_dictionary:
        patient_dictionary[patient_name].append(filename)

    else:
        patient_dictionary[patient_name] = [filename]

#print("Finished Organizing into dicitonary")

#print("Short Preview of the Dictionary:")

def read_csv_to_dataframe(file_path):
    try:
        # Read the CSV file into a Pandas DataFrame
        dataframe = pd.read_csv(file_path)

        # Return the DataFrame
        return dataframe

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
patient_data_dictionary = {}
#read the csv for all patient files
for patient in patient_dictionary:
    files = patient_dictionary[patient]
    if len(files) != 2:
        continue
    patient_data_dictionary[patient] = {'events': read_csv_to_dataframe(path + files[0]),'ppg':read_csv_to_dataframe(path + files[1])}



#Change the time from unix format to regular date

for patient in patient_data_dictionary:

    patient_data_dictionary[patient]['ppg']['ts-datetime'] = pd.to_datetime(patient_data_dictionary[patient]['ppg']['ts'], unit='s')
    patient_data_dictionary[patient]['events']['start-datetime'] = pd.to_datetime(patient_data_dictionary[patient]['events']['start'], unit='s')
    patient_data_dictionary[patient]['events']['end-datetime'] = patient_data_dictionary[patient]['events']['start-datetime'] + pd.to_timedelta(patient_data_dictionary[patient]['events']['duration'], unit='s')


def getIndices(df):
    return df.index.values

def flattenedList(list_of_lists):
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return flattened_list

def consecutive_lists(sorted_list):
    result = []
    current_list = []

    for num in sorted_list:
        if not current_list or num == current_list[-1] + 1:
            current_list.append(num)
        else:
            result.append(current_list)
            current_list = [num]

    if current_list:
        result.append(current_list)

    return result

def randomlySampleList(lst,k):
    try:
        return random.sample(lst,k=k)
    
    except:
        return lst
    
def randomlySamplePpg_Df(df:pd.DataFrame,time:int) -> pd.DataFrame:
    try:
        first = df['ts-datetime'].values[0]
        last = df['ts-datetime'].values[-1]

        random_start_time = pd.to_datetime(random.uniform(first, last-pd.to_timedelta(time,unit = 's')), unit='s')
        window = df[(df['ts-datetime']>= random_start_time) & (df['ts-datetime']<= random_start_time + pd.to_timedelta(10,unit = 's'))]
        return window
    
    except:
        return df
    

sample_time = 15
for patient in patient_data_dictionary:
    print(f"Processing Patient {patient}")
    allApneaPpg_dfs = []
    allEventPpg_dfs = []
    allNonEventPpg_dfs = []

    dictionary = patient_data_dictionary[patient]
    ppg = dictionary['ppg']
    events = dictionary['events']

    for index,event in events.iterrows():
        inEventCondition = (ppg['ts-datetime'] <= event['end-datetime']) & (ppg ['ts-datetime'] >= event['start-datetime'])
        eventPpg = ppg[inEventCondition]

        if event['name'] in apnea_labels:
            allApneaPpg_dfs.append(eventPpg)
        #eventPpg = randomlySamplePpg_Df(eventPpg,sample_time)
        allEventPpg_dfs.append(eventPpg)

    allEventPpg_Indices = sorted(list(set(flattenedList(list(map(getIndices,allEventPpg_dfs))))))
    
    nonEventPpg_Indices = ppg[~ppg.index.isin(allEventPpg_Indices)].index.values

    sample_size = len(allApneaPpg_dfs)
    list_Of_NonEventPpgIndices_Lists = consecutive_lists(nonEventPpg_Indices)

    randomlySampledNonEvents = randomlySampleList(list_Of_NonEventPpgIndices_Lists,sample_size)
    for indices in randomlySampledNonEvents:
        nonEventPpg = ppg.iloc[indices]
        nonEventPpg = randomlySamplePpg_Df(nonEventPpg,sample_time)
        allNonEventPpg_dfs.append(nonEventPpg)
    print(f"Processed {len(allApneaPpg_dfs)} Apnea Events")
    print(f"Processed {len(randomlySampledNonEvents)} non-Events")
    apnea_fileName = 'apnea_ppg.pkl'
    nonEvent_fileName = 'nonEvent_ppg.pkl'
    os.makedirs(f"./samples/{patient}/",exist_ok=True)

    if(len(allApneaPpg_dfs) > 0):
        with open(f"./samples/{patient}/{apnea_fileName}",'wb') as pickle_file:
            pickle.dump(allApneaPpg_dfs,pickle_file)

        with open(f"./samples/{patient}/{nonEvent_fileName}",'wb') as pickle_file:
            pickle.dump(allNonEventPpg_dfs,pickle_file)

    