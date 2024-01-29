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
import heartpy as hp
apnea_labels = ["Mixed Apnea", "Central Apnea", "Obstructive Apnea"]

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

matrix = np.empty ((0,10))
sample_rate = 64

dummy = patient_data_dictionary
for patient in dummy:
    events = dummy[patient]['events']
    ppg_signal = dummy[patient]['ppg']
    apnea = events[events['name'].isin(apnea_labels)]
    ppg_signal.set_index('ts-datetime', inplace=True)

    # Define the frequency for the intervals (e.g., '10S' for 10-second intervals)
    freq = '10S'

    # Group the DataFrame by the specified frequency
    ppg_10s_intervals = ppg_signal.groupby(pd.Grouper(freq=freq))

    # Now ppg_10s_intervals contains the DataFrame divided into 10-second intervals
    # You can access individual groups using the get_group method
    for interval, data in ppg_10s_intervals:
        ppg = data
        overlapping_apnea = apnea[(apnea['start-datetime'] <= interval  + pd.to_timedelta(10,unit = 's')) & (apnea['end-datetime'] >=interval)]
        # ppg['ch2'] = (ppg['ch2'] - ppg['ch2'].min(axis = 0))/(ppg['ch2'].max(axis=0)-ppg['ch2'].min(axis=0))
        ppg = ppg['ch2']
        try:
            label = 0
            wd,m = hp.process(ppg,sample_rate = sample_rate)

            #['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']
        # Basic metrics
            print(m)
            heart_rate = m['bpm']  # Heart Rate in BPM
            ibi = m['ibi']  # Average Interbeat Interval
            sdnn = m['sdnn']  # Standard Deviation of NN intervals
            sdsd = m['sdsd']  # Standard Deviation of successive differences between adjacent NN intervals

            # HRV metrics
            rmssd = m['rmssd']  # Root Mean Square of Successive Differences
            pnn50 = m['pnn50']  # proportion of differences between adjacent NN intervals that are greater than 50ms

            std = np.std(ppg)
            mean = np.mean(ppg)
            var = np.var(ppg)



            if not overlapping_apnea.empty:

                label = 1

            feature_row = np.array([heart_rate,ibi,sdnn,sdsd,rmssd,pnn50,std,mean,var,label])
            feature_row = np.nan_to_num(feature_row,nan=-1)
            matrix = np.vstack([matrix,feature_row])



        except hp.exceptions.BadSignalWarning as e:
            pass
            #print("Bad signal quality. Consider checking the data or further preprocessing.")


        except:
            print('error')


pd.DataFrame(matrix).to_csv('./matrix.csv',index = True)


def plotFeatures(labels,features,fileName,featureName):
    plt.scatter(x = labels, y = features)
    plt.title(featureName)
    plt.ylabel(featureName)
    plt.xlabel('apnea = 1, non-apnea = 0')
    plt.savefig(fileName)
    plt.clf()


#plotting heart rate
labels = matrix[:,9]

featureNames = ['Heart Rate (BPM)','IBI','SDNN','SDSD','RMSSD','PNN50','STDS','MEANS','VARS']

for i in range(len(featureNames)):
    featureName = featureNames[i]

    features = matrix[:,i]

    plotFeatures(labels,features,f"./visualizations/{featureName}.png",featureName)


