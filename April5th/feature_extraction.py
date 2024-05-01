PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
SAMPLING_RATE = 100
import argparse
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
EDF_COLUMNS = ['time', 'zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI',\
       'ppgRaw', 'ppgFiltered', 'ppgSQI', 'x', 'y', 'z', 'chestTemp',\
       'limbTemp', 'x_detrend', 'z_detrend', 'y_detrend', 'PAT', 'PATmedian',\
       'PATdetrend', 'footpeak', 'footpeakmedian', 'HRmedian', 'HR', 'SpO2',\
       'RR', 'RRsqi', 'PPGamp', 'PPGampmedian', 'sleepstage', 'arousal_events',\
       'resp_events', 'PLM_events', 'SpO2_events']

COLUMNS_OF_INTEREST = ['ppgFiltered','x_detrend','y_detrend','z_detrend','SpO2','HR']

def normalize(data):
    data_normalized = (data - data.min()) / (data.max() - data.min())
    return data_normalized
def roundToFirstdigit(num):
   first_2_digits = float(str(num)[0:2])
   returnVal = first_2_digits/10
   return round(returnVal)

def get_Paths_and_Names(path = PATH):
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

def extractFeatures(path_and_name,window_time,window_move_time):
    path = path_and_name['path']
    name = path_and_name['name']
    dict = {
        'name':name,
    }
    features_and_label = {
        'label':[]
    }
    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    start = 0
    end = SAMPLING_RATE * window_time
    increment = SAMPLING_RATE* window_move_time
    while(end < len(df)):
        current_sample = df.iloc[start:end,:]
        resp_events = current_sample['resp_events'].values
        resp_events = np.array(list(map(roundToFirstdigit,resp_events)))
        any_positive = np.any(resp_events > 0)
        label = 0
        if (any_positive):
            label = 1
        features_and_label['label'].append(label)

        for column in COLUMNS_OF_INTEREST:
            if not f'{column}_mean' in features_and_label:
                features_and_label[f'{column}_mean'] = []

            if not f'{column}_var' in features_and_label:
                features_and_label[f'{column}_var'] = []

            if not f'{column}_std' in features_and_label:
                features_and_label[f'{column}_std'] = []
            data = df[column].values
            normalized = normalize(data)

            mean = np.mean(normalized)
            std = np.std(normalized)
            var = np.var(normalized)

            features_and_label[f'{column}_mean'].append(mean)
            features_and_label[f'{column}_var'].append(var)
            features_and_label[f'{column}_std'].append(std)

        start,end = start + increment,end+increment

    return pd.DataFrame(features_and_label)



    



if __name__ == "__main__":
    t0 = datetime.now()
    print(f"Job started at {t0}",flush = True)
    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes = npus)
    paths_and_names = get_Paths_and_Names()
    parser = argparse.ArgumentParser(description='script to extract features from varying times of data')

    parser.add_argument('window_time_in_seconds', type=int, help='A sample integer argument')
    parser.add_argument('time_movement_in_seconds',type=int, help = 'Time that window moves')
    # Parse the arguments
    args = parser.parse_args()
    window_time = args.window_time_in_seconds
    window_move_time = args.time_movement_in_seconds

    results = [pool.apply_async(extractFeatures,args = (x,window_time,window_move_time)) for x in paths_and_names]
    all_features = [p.get() for p in results]
    pool.close()
    pool.join()

     # Serialize and write to file
    with open('./features_and_targets.pkl', 'wb') as f:
        pickle.dump(all_features, f)

    t1 = datetime.now()

    print(f'Job ended at {t1}',flush = True)
    print(f"Duration was {t1-t0}",flush = True)


