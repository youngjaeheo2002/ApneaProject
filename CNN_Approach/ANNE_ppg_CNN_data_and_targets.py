ANNE_PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215/'
ANNE_SAMPLING_RATE = 100
SAMPLE_LENGTH_IN_SECONDS = 10
import mne
import numpy as np
import multiprocessing as mp
import os
import pickle
'''
the features will the ppg_signals themselves
'''

from pathlib import Path 

def fill_nans_with_nearest(arr):
    """
    Fills NaN values in a numpy array with the nearest valid value.
    If the NaN is at the beginning of the array, it uses the next valid value.
    """
    # Check if the first element is NaN and there's at least one more element
    if np.isnan(arr[0]) and len(arr) > 1:
        arr[0] = arr[np.where(~np.isnan(arr))[0][0]]

    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]
    return arr

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

def separate_Data_by_Respiratory_Event(path_and_name, addedTimeToTheSide):
  path = path_and_name['path']
  name = path_and_name['name']

  raw = mne.io.read_raw_edf(path)
  df = raw.to_data_frame()
  add_Indices = addedTimeToTheSide * 1000
  returnDict = {
    'name':name,
    "data":[],
    "target":[]
  }

  start = 0
  end = 0
  length = len(df)


  while(end  != length):

    current_Label = roundToFirstdigit(df.iloc[start]['resp_events'])
    end_Label = roundToFirstdigit(df.iloc[end]['resp_events'])
    if (current_Label != end_Label):

      event_Period = df.iloc[max(start-add_Indices,0):min(end+add_Indices,length-1)]

      returnDict["data"].append(event_Period['ppgFiltered'].values)
      returnDict["target"].append(current_Label)
      start = end
    end += 1

  event_Period = df.iloc[max(start-add_Indices,0):min(end+add_Indices,length-1)]

  returnDict["data"].append(event_Period['ppgFiltered'].values)
  returnDict["target"].append(current_Label)

  return create10SecondSnippets(returnDict)

def roundToFirstdigit(num):
   first_2_digits = float(str(num)[0:2])
   returnVal = first_2_digits/10
   return round(returnVal)

def normalize_data_np(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def separateInto10SecondSnippets(data,target):
    sample_length = SAMPLE_LENGTH_IN_SECONDS * ANNE_SAMPLING_RATE  
    start = 0
    end = sample_length 
    length = len(data)
    res = {
       'target':np.empty((0,1)),
       'data':np.empty((0,sample_length))
    }
    while end < length:
      res['data'] = np.vstack((res['data'],normalize_data_np( fill_nans_with_nearest(data[start:end]))))
      res['target'] = np.vstack((res['target'],np.array([target])))
      start = end
      end += sample_length

    return res

    

    
def create10SecondSnippets(separated):
    sample_length = SAMPLE_LENGTH_IN_SECONDS * ANNE_SAMPLING_RATE
    name = separated['name']
    res = {
       'name':name,
       'target':np.empty((0,1)),
       'data':np.empty((0,sample_length))
    }

    data = separated['data']
    target = separated['target']

    for i in range(len(data)):
      current_data = data[i]
      current_target = target[i]
      snippets = separateInto10SecondSnippets(current_data,current_target)
      res['target'] = np.vstack((res['target'],snippets['target']))
      res['data'] = np.vstack((res['data'],snippets['data']))

    return res

if __name__ == "__main__":
    
    paths_and_names = get_Paths_and_Names()

    # paths_and_names = [{
    #    'path':"/scratch/alim/overnight_validation/ANNE-PSG231215/20-08-20-21_33_30.C1442.L1215.185/20-08-20-21_33_30.C1442.L1215.185-features.edf",
    #    'name':'test'
    # }]
    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes = npus)
    # print(paths_and_names)

    separated = [pool.apply_async(separate_Data_by_Respiratory_Event,args = (x,0)) for x in paths_and_names]
    separated = [p.get() for p in separated]
    pool.close()
    pool.join()

    # Serialize and write to file
    with open('./data_and_targets1.pkl', 'wb') as f:
        pickle.dump(separated, f)



    # with open("separated1.txt",'w') as f:
    #    f.write(str(separated))

    # print(separated[0]['data'].shape,separated[0]['target'].shape)


    
