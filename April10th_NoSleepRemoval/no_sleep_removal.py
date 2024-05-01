PATH = '/scratch/alim/overnight_validation/ANNE-PSG231215'
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

def round_to_nearest_million(x):
    return round(x / 1000000) * 1000000

def find_next_sleep_period(start,values):
    while(values[start] == 0):
        start = start + 1

    end = start
    while(values[end] != 0 and end < len(values)):
        end += 1

    return start,end
def extract_sleep_periods(path_and_name):
    path = path_and_name['path']
    name =path_and_name['name']
    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    df['sleepstage_rounded'] = df['sleepstage'].apply(round_to_nearest_million)
    #print(np.unique(df['sleepstage_rounded'].values))

    arr = df['sleepstage_rounded'].values

    # Create a boolean mask where True represents the elements > 0
    mask = arr > 0

    # Find the differences in the mask to identify section boundaries
    diff = np.diff(mask.astype(int))

    # Start indices of positive sections
    start_indices = np.where(diff == 1)[0] + 1

    # If the first element is positive, prepend a start index for it
    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)

    # End indices of positive sections
    end_indices = np.where(diff == -1)[0]

    # If the last element is positive, append an end index for it
    if mask[-1]:
        end_indices = np.append(end_indices, len(arr) - 1)

    # Extracting the sections
    positive_sections = [df.iloc[start:end+1] for start, end in zip(start_indices, end_indices)]

    # Display the sections
    return positive_sections





if __name__ == '__main__':
    t0 = datetime.now()
    print(f'non-sleep removal started at {t0}',flush = True)
    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes=npus)
    paths_and_names = get_Paths_and_Names()

    results = [pool.apply_async(extract_sleep_periods,args = (x,)) for x in paths_and_names]
    results = [p.get() for p in results]
    pool.close()
    pool.join()

    with open('./ANNE_only_sleep.pkl', 'wb') as f:
        pickle.dump(results, f)

    t1 = datetime.now()

    print(f'Job ended at {t1}',flush = True)
    print(f"Duration was {t1-t0}",flush = True)

