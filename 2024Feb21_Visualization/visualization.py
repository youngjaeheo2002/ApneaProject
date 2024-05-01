import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
PLOTS_PATH = './PLOTS4'
AMPLITUDE = 1000

COLOR_MAPPING = {
   0:'green',
   1:'red',
   2:'purple',
   3:'blue',
   4:'orange',
   5:'black'
}

ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
COLUMNS_OF_INTEREST = ['ecgRaw','ecgProcessed','ppgFiltered','x', 'y', 'z','SpO2','HR']
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

def getColoursForLabels(labels):
    colors = []

    for label in labels:
       colors.append(COLOR_MAPPING[label])

    return colors

def transform_Values(values, offset, amplitude):
   values_min = values.min()
   values_max = values.max()

   new_min = offset
   new_max = offset + amplitude

   normalized_data = new_min + ((values - values_min)*(new_max - new_min)/(values_max - values_min))

   return normalized_data


def find_label_transitions_with_labels(labels):
    """
    Finds the indices where one label ends and another begins, along with
    the labels before and after the transition.
    
    :param labels: A list or NumPy array of labels.
    :return: A dictionary with transition indices as keys and tuples of 
             (label_before, label_after) as values.
    """
    
    # Find indices where the value changes
    change_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    
    # Gather the labels before and after each transition
    transitions = {
        idx: (labels[idx - 1], labels[idx])
        for idx in change_indices
    }
    with open('transitions.txt','a') as file:
       file.write(str(transitions)+'\n')

    return transitions

def plot_Transitions(transitions,times):
    times_ticks = []
    times_ticks_labels = []
    for idx, (before, after) in transitions.items():
        time = times[idx]
        times_ticks.append(time)
        times_ticks_labels.append(f"{before}->{after}")
        plt.axvline(x = time,color = 'red',linestyle= '--', label = "Separator")
    plt.xticks(times_ticks,times_ticks_labels, fontsize = 16)


def plot_Data(data,label,name,index):
    plt.clf()
    plt.figure(figsize = (30,10))
    plotFileName = f"{PLOTS_PATH}/{name}/{label}/{index}.png"
    os.makedirs(f"{PLOTS_PATH}/{name}/{label}/",exist_ok=True)
    times = data['time'].values
    offset = 0
    labels = np.array(list(map(roundToFirstdigit,data['resp_events'])))
    transitions = find_label_transitions_with_labels(labels)
    y_tick_locations = []
    for column in COLUMNS_OF_INTEREST:
       values = data[column]
       transformed_Values = transform_Values(values = values,offset = offset, amplitude=AMPLITUDE)
       plt.plot(times,transformed_Values)
       y_tick_locations.append(offset + (AMPLITUDE/2))
       offset += AMPLITUDE
       
    y_ticks = COLUMNS_OF_INTEREST
    plt.yticks(y_tick_locations,y_ticks,fontsize = '36')
    plot_Transitions(transitions=transitions,times = times)
    plt.savefig(plotFileName)
    plt.clf()



def plot_Separated(dict,name):
    for label in dict:
        datas = dict[label]
        for i in range (len(datas)):
            data = datas[i]
            plot_Data(data = data, label = label, name = name,index = i)


def visualizePatient(path_and_name):
    path = path_and_name['path']
    name = path_and_name['name']

    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    # print(f"COLUMNS COLUMNS COLUMNS{df.columns}")
    separated = separate_Data_by_Respiratory_Event(df,10)

    plot_Separated(dict = separated,name = name)

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

    results = [pool.apply_async(visualizePatient,args = (x,)) for x in paths_and_names]
    for result in results:
        result.get()
    pool.close()
    pool.join()

    end = datetime.now()
    print(f"Job ended at {end}")
    duration = end - start
    print(f"Job took {duration} long")