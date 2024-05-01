#!/usr/bin/env python
# coding: utf-8

# In[18]:


#import librararies
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import nolds
import glob
import mne
import multiprocessing as mp 
import os
from datetime import datetime
start = datetime.now()
print(f"Started job at {start}")



# In[3]:


ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
'''
 Numeric coding of repiratory events
   > 1 - "RERA"; 2 - "Hypopnea"; 3 - "Obstructive Apnea"; 4 - "Central Apnea"; 5 - "Mixed Apnea"
'''
#features to extract: mean, std, var for all channels
SAMPLING_RATE = 256
DATA_COLUMNS = ['zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI',\
       'ppgRaw', 'ppgFiltered', 'ppgSQI', 'x', 'y', 'z', 'chestTemp',\
       'limbTemp', 'x_detrend', 'z_detrend', 'y_detrend', 'PAT', 'PATmedian',\
       'PATdetrend', 'footpeak', 'footpeakmedian', 'HRmedian', 'HR', 'SpO2',\
       'RR', 'RRsqi', 'PPGamp', 'PPGampmedian', 'sleepstage']

features_Df = pd.DataFrame()



# In[4]:


#get metadata csv
metadata_file_name = ANNE_PATH + "/ANNE-PSG_metadata.csv"
metadata_df = pd.read_csv(metadata_file_name)
metadata_df.head()
print(metadata_df[metadata_df['file'] == '23-11-01-19_35_59.C3884.L3802.696'])


# In[13]:


def separate_Data_by_Respiratory_Event(df):
  returnDict = {
    0:[],1:[],2:[],3:[],4:[],5:[]
  }

  start = 0
  end = 0
  length = len(df)


  while(end  != length):

    current_Label = int(str(df.iloc[start]['resp_events'])[0])
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




# In[14]:


from pathlib import Path
import mne
parent_dir = Path(ANNE_PATH)
features = pd.DataFrame()

ncpus= int(os.environ.get('SLURM_CPUS_PER_TASK',default = 1))
pool = np.Pool(processes = ncpus)

#iterate through every patient in ANNE-PSG231215
for subDir in parent_dir.iterdir():
    if subDir.is_dir():
        for entry in subDir.iterdir():
            absolute_entry_path = str(entry)

            raw = mne.io.read_raw_edf(absolute_entry_path)
            df = raw.to_data_frame()

            separated = separate_Data_by_Respiratory_Event(df)
            for label in separated:
                for data in separated[label]:

                    feature_row = {}
                    mean_std_var = extract_Mean_Std_Var(df = data)
                    feature_row.update(mean_std_var)

                    feature_row['event'] = label

                    feature_row['name'] = subDir.name
                    feature_row = pd.DataFrame(feature_row)
                    features = pd.concat([features,feature_row],ignore_index = True)

end = datetime.now()

features.to_csv(f"ANNE_FEATURES_{end}_.csv")

                #print(separate_Data_by_Respiratory_Event(df))




