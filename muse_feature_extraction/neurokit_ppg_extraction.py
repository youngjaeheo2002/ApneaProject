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
MUSE_SAMPLING_RATE = 64

'''
THE FOLLOWING LENGTH OF CODE LINES 17-83 PROCESSES FILES INTO A USEABLE PATIENT_DATA_DIRECTORY
'''
def read_files_in_directory(directory_path):
    try:
        #print("Finding all MUSE-PSG files")
        # Get a list of all files in the directory
        file_list = sorted([entry.name for entry in os.scandir(directory_path) if
                    entry.is_file() and any(keyword in entry.name for keyword in ['ppg','events'])])[30:33]


                     
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
    patient_data_dictionary[patient] = [read_csv_to_dataframe(path + files[0]),read_csv_to_dataframe(path + files[1])]



#Change the time from unix format to regular date

for patient in patient_data_dictionary:

    patient_data_dictionary[patient][1]['ts-datetime'] = pd.to_datetime(patient_data_dictionary[patient][1]['ts'], unit='s')
    patient_data_dictionary[patient][0]['start-datetime'] = pd.to_datetime(patient_data_dictionary[patient][0]['start'], unit='s')
    patient_data_dictionary[patient][0]['end-datetime'] = patient_data_dictionary[patient][0]['start-datetime'] + pd.to_timedelta(patient_data_dictionary[patient][0]['duration'], unit='s')
    
patient_data_dictionary['2020-08-17T211411-0400_5007-ELYP-1F41'][2].head()
 

 