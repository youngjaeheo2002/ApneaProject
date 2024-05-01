import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PLOTS_PATH = './PLOTS4'

COLOR_MAPPING = {
   0:'green',
   1:'red',
   2:'purple',
   3:'blue',
   4:'orange',
   5:'black'
}

ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
COLUMNS_OF_INTEREST = ['ecgRaw','ecgProcessed','ppgFiltered','x', 'y', 'z']

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

def get_Colors_For_Labels(labels):
   colors = map(lambda x: COLOR_MAPPING[x],labels)
   return list(colors)

def scaleValuesBetweenLowerAndUpper(values,lower = 0,upper = 1000):
   values_min = values.min()
   values_max = values.max()

   normalized_values = (values - values_min)/(values_max -values_min)
   return normalized_values * (upper - lower) + lower


def plot_3_seconds_before_and_3seconds_after(data,label,name,index):
    plt.clf()
    plt.figure(figsize=(1920,1080))
    before = data['before']
    event = data['event']
    after = data['after']
    before_times = before['time'].values
    event_times = event['time'].values
    after_times = after['time'].values
    plt.yticks([500,1500,2500,3500,4500,5500],COLUMNS_OF_INTEREST)
    offset = 0  
    for column in COLUMNS_OF_INTEREST:
       #before
       before_column = before[column].values
       before_labels = before['resp_events']
       before_labels = list(map(roundToFirstdigit,before_labels))
       before_colors = get_Colors_For_Labels(before_labels)
       #after
       after_column = after[column].values
       after_labels= after['resp_events']
       after_labels = list(map(roundToFirstdigit,after_labels))
       after_colors = get_Colors_For_Labels(after_labels)
       #event
       event_column = event[column].values
       event_labels = event['resp_events']
       event_labels = list(map(roundToFirstdigit,event_labels))
       event_colors = get_Colors_For_Labels(event_labels)

       all_values = np.concatenate((before_column,event_column,after_column))
       all_times = np.concatenate((before_times,event_times,after_times))
       all_colors = np.concatenate((before_colors,event_colors,after_colors))
       scaled_values = scaleValuesBetweenLowerAndUpper(lower = 0,upper = 1000, values = all_values)
       
       for i in range(len(all_values)):
          plt.plot(all_times[i],scaled_values[i] + offset, color = all_colors[i])
       offset += 1000

    os.makedirs(f"{PLOTS_PATH}/{name}/{label}/",exist_ok= True)
    plt.savefig(f"{PLOTS_PATH}/{name}/{label}/{index}.png")
    plt.clf()


def roundToFirstdigit(num):
   first_2_digits = float(str(num)[0:2])
   returnVal = first_2_digits/10
   return round(returnVal)

def separate_Data_by_Respiratory_Event_and_add_3_seconds_before_and_3seconds_after(df):
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
      before_Period = df.iloc[max(0,start-3000):start]
      after_Period = df.iloc[end:min(end+3000,length)]
      event_Period = df.iloc[start:end]
      if not current_Label in returnDict:
        
        
        returnDict[current_Label] = [
        ]

      returnDict[current_Label].append({
         'before':before_Period,
         'event':event_Period,
         'after':after_Period
      })

      start = end
    end += 1
  before_Period = df.iloc[max(0,start-3000):start]
  after_Period = df.iloc[end:min(end+3000,length)]
  event_Period = df.iloc[start:end]


  if not current_Label in returnDict:
    
    returnDict[current_Label] = [
    ]

  returnDict[current_Label].append({
         'before':before_Period,
         'event':event_Period,
         'after':after_Period
      })

  return returnDict

def plotFeatures_3_seconds_before_and_after_For_all(path_and_name):
    features = pd.DataFrame()
    path = path_and_name['path']
    name = path_and_name['name']

    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()

    separated = separate_Data_by_Respiratory_Event_and_add_3_seconds_before_and_3seconds_after(df)
    for label in separated:
        lst = separated[label]

        for i in range(len(lst)):
            data = lst[i]
            plot_3_seconds_before_and_3seconds_after(data,label,name,i)
          
          
          
if __name__ == '__main__':
    pass
    start = datetime.now()
    print(f"Job started at {start}")

    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes = npus)
    paths_and_names = get_Paths_and_Names()

    paths_and_names = [{
       'path':"/scratch/alim/overnight_validation/ANNE-PSG231215/20-08-20-21_33_30.C1442.L1215.185/20-08-20-21_33_30.C1442.L1215.185-features.edf",
       'name':'test'
    }]


    # for path_and_and_name in paths_and_names:
    #    plotFeatures_3_seconds_before_and_after_For_all(path_and_name=path_and_and_name)

    results = [pool.apply_async(plotFeatures_3_seconds_before_and_after_For_all,args = (x,)) for x in paths_and_names]
    for result in results:
        result.get()
    pool.close()
    pool.join()

    end = datetime.now()
    print(f"Job ended at {end}")
    duration = end - start
    print(f"Job took {duration} long")