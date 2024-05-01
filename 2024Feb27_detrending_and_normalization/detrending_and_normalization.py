import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.signal import detrend

PLOTS_PATH = './PLOTS4'
AMPLITUDE = 1000

ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
COLUMNS_OF_INTEREST = ['ecgRaw','ecgProcessed','ppgFiltered','x', 'y', 'z','SpO2','HR']
COLUMNS_TO_DETREND = ['ecgRaw','ecgProcessed','ppgFiltered','SpO2','HR']
COLUMNS_ALREADY_DETRENDED = ['x_detrend','y_detrend','z_detrend']
def roundToFirstdigit(num):
   first_2_digits = float(str(num)[0:2])
   returnVal = first_2_digits/10
   return round(returnVal)

def separate_Data_by_Respiratory_Event(df, addedTimeToTheSide):
  add_Indices = addedTimeToTheSide * 1000
  returnDict = {
    0:[],1:[],2:[],3:[],4:[],5:[]
  }

  start = 0
  end = 0
  length = len(df)


  while(end  != length):

    current_Label = roundToFirstdigit(df.iloc[start]['resp_events'])
    end_Label = roundToFirstdigit(df.iloc[end]['resp_events'])
    if (current_Label != end_Label):

      event_Period = df.iloc[max(start-add_Indices,0):min(end+add_Indices,length-1)]
      if not current_Label in returnDict:
        
        
        returnDict[current_Label] = [
        ]

      returnDict[current_Label].append(event_Period)

      start = end
    end += 1

  event_Period = df.iloc[max(start-add_Indices,0):min(end+add_Indices,length-1)]

  if not current_Label in returnDict:
    
    returnDict[current_Label] = [
    ]

  returnDict[current_Label].append(event_Period)

  return returnDict


def get_Paths_and_Names(path = ANNE_PATH):
    paths_and_names = []
    parent_dir = Path(path)

    for subDir in parent_dir.iterdir():
        if subDir.is_dir():
            for entry in subDir.iterdir():
                absolute_entry_path = str(entry)
                paths_and_names.append({
                    'path': absolute_entry_path,
                    'name':subDir.name
                })


    return paths_and_names

def detrendSeparated(separated):
    separated = separated.copy()
    for label in separated:
        labeled_data = separated[label]

        for data in labeled_data:

            for column in COLUMNS_TO_DETREND:
                data[f"{column}_detrend"] = data[f"{column}"].diff()

                #now normalize

            for column in COLUMNS_OF_INTEREST:
               data[f"{column}_normalized_and_detrend"] = data[f"{column}_detrend"]





    return separated

def identify_PQRST_waves():
  pass

def extractFeaturesHelper(separated,name):
    dict = {
       'label':[],
       'name':[]
    }

    separated = separated.copy()

    for label in separated:
      label_data = separated[label]

      for data in label_data:
        dict['label'].append(label)
        dict['name'].append(name)
          
        #data that needs to be detrended AND normalized
        for column in COLUMNS_TO_DETREND:
          #detrend values first
          values = data[column].values
          detrended_values = detrend(values)

          #next normalize the data between 0 and 1
          normalized_values_0_1 = (detrended_values - detrended_values.min())/(detrended_values.max() - detrended_values.min())

          normalized_values_minus1_1 = normalized_values_0_1 * 2 -1

          mean = np.mean(normalized_values_minus1_1)
          std = np.std(normalized_values_minus1_1)
          var = np.var(normalized_values_minus1_1)

          if not f"{column}_mean" in dict:
            dict[f"{column}_mean"] = []

          if not f"{column}_var" in dict:
            dict[f"{column}_var"] = []

          if not f"{column}_std" in dict:
            dict[f"{column}_std"] = []

          dict[f"{column}_mean"].append(mean)
          dict[f"{column}_var"].append(var)
          dict[f"{column}_std"].append(std)

          
          
          
            
        #data that is already detrended only needs to be normalized
        for column in COLUMNS_ALREADY_DETRENDED:
          values = data[column].values

          normalized_values_0_1 = (values-values.min())/(values.max()-values.min())

          normalized_values_minus1_1 = normalized_values_0_1 * 2 -1

          mean = np.mean(normalized_values_minus1_1)
          std = np.std(normalized_values_minus1_1)
          var = np.var(normalized_values_minus1_1)

          if not f"{column}_mean" in dict:
            dict[f"{column}_mean"] = []

          if not f"{column}_var" in dict:
            dict[f"{column}_var"] = []

          if not f"{column}_std" in dict:
            dict[f"{column}_std"] = []

          dict[f"{column}_mean"].append(mean)
          dict[f"{column}_var"].append(var)
          dict[f"{column}_std"].append(std)
          
    return_Df = pd.DataFrame(dict)

    return return_Df


def extractFeatures(path_and_name):
    path = path_and_name['path']
    name = path_and_name['name']

    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    # print(f"COLUMNS COLUMNS COLUMNS{df.columns}")
    separated = separate_Data_by_Respiratory_Event(df,3)

    features = extractFeaturesHelper(separated,name)

    return features
if __name__ == '__main__':
    pass
    start = datetime.now()
    print(f"Job started at {start}")

    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes = npus)
    paths_and_names = get_Paths_and_Names()

    # paths_and_names = [{
    #    'path':"/scratch/alim/overnight_validation/ANNE-PSG231215/20-08-20-21_33_30.C1442.L1215.185/20-08-20-21_33_30.C1442.L1215.185-features.edf",
    #    'name':'test'
    # }]




    # for path_and_and_name in paths_and_names:
    #    plotFeatures_3_seconds_before_and_after_For_all(path_and_name=path_and_and_name)

    results = [pool.apply_async(extractFeatures,args = (x,)) for x in paths_and_names]
    all_features = [p.get() for p in results]
    pool.close()
    pool.join()

    final_df = pd.concat(all_features)
    final_df.to_csv('features2.csv')
    end = datetime.now()
    print(f"Job ended at {end}")
    duration = end - start
    print(f"Job took {duration} long")