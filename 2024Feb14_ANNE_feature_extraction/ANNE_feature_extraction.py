import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd



ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
DATA_COLUMNS = ['zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI',\
       'ppgRaw', 'ppgFiltered', 'ppgSQI', 'x', 'y', 'z', 'chestTemp',\
       'limbTemp', 'x_detrend', 'z_detrend', 'y_detrend', 'PAT', 'PATmedian',\
       'PATdetrend', 'footpeak', 'footpeakmedian', 'HRmedian', 'HR', 'SpO2',\
       'RR', 'RRsqi', 'PPGamp', 'PPGampmedian', 'sleepstage']


def cube(x):
    return x**3

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

def separate_Data_by_Respiratory_Event(df):
  returnDict = {
    0:[],1:[],2:[],3:[],4:[],5:[]
  }

  start = 0
  end = 0
  length = len(df)


  while(end  != length):

    current_Label = df.iloc[start]['resp_events']
    end_Label = int(str(df.iloc[end]['resp_events'])[0])
    if (current_Label != end_Label):

      event_Period = df.iloc[start:end]
      if not current_Label in returnDict:
        
        
        returnDict[current_Label] = [
        ]

      returnDict[current_Label].append(event_Period)

      start = end
    end += 1

  event_Period = df.iloc[start:end]

  if not current_Label in returnDict:
    
    returnDict[current_Label] = [
    ]

  returnDict[current_Label].append(event_Period)

  return returnDict
  
def extract_Mean_Std_Var(df, desired_columns = DATA_COLUMNS):
  #returns dictionary with feature
  returnDict = {}
  for column in desired_columns:
    values = df[column].values
    returnDict[f"{column}_mean"] = [np.mean(values)]
    returnDict[f"{column}_std"] = [np.std(values)]
    returnDict[f"{column}_var"] = [np.var(values)]


  return returnDict

def get_Features_For_Patient(path_and_name):
    features = pd.DataFrame()
    path = path_and_name['path']
    name = path_and_name['name']

    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()

    separated = separate_Data_by_Respiratory_Event(df)

    for label in separated:
        for data in separated[label]:

            feature_row = {}
            mean_std_var = extract_Mean_Std_Var(df = data)
            feature_row.update(mean_std_var)

            feature_row['event'] = label

            feature_row['name'] = name
            feature_row = pd.DataFrame(feature_row)
            features = pd.concat([features,feature_row],ignore_index = True)
    return features


if __name__ == '__main__':

    start = datetime.now()
    print(f"Job started at {start}")

    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes = npus)
    paths_and_names = get_Paths_and_Names()
    '''
    TESTING FOR ONE FILE
    paths_and_names = [{
       'path':"/scratch/alim/overnight_validation/ANNE-PSG231215/20-08-20-21_33_30.C1442.L1215.185/20-08-20-21_33_30.C1442.L1215.185-features.edf",
       'name':'test'
    }]
    '''
    all_features = [pool.apply_async(get_Features_For_Patient,args = (x,)) for x in paths_and_names]
    all_features = [p.get() for p in all_features]
    all_features = pd.concat(all_features,ignore_index = True)

    all_features.to_csv(f"ANNE_Feature_Extraction_{start}.csv")
    '''
    data = [1, 2, 3, 4, 5, 6]
    results = [pool.apply_async(cube, args=(x,)) for x in data]
    cubes = [p.get() for p in results]
    print(cubes)
    '''


    end = datetime.now()
    print(f"Job ended at {end}")

