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
import neurokit2 as nk
from scipy.signal import find_peaks
import scipy
from scipy.signal import detrend
SAMPLING_RATE = 100
SCRATCH = '/home/heoyoun1/scratch'
EDF_COLUMNS = ['time', 'zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI',\
       'ppgRaw', 'ppgFiltered', 'ppgSQI', 'x', 'y', 'z', 'chestTemp',\
       'limbTemp', 'x_detrend', 'z_detrend', 'y_detrend', 'PAT', 'PATmedian',\
       'PATdetrend', 'footpeak', 'footpeakmedian', 'HRmedian', 'HR', 'SpO2',\
       'RR', 'RRsqi', 'PPGamp', 'PPGampmedian', 'sleepstage', 'arousal_events',\
       'resp_events', 'PLM_events', 'SpO2_events']

COLUMNS_OF_INTEREST = ['zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI',\
       'ppgRaw', 'ppgFiltered', 'ppgSQI', 'x', 'y', 'z', 'chestTemp',\
       'limbTemp', 'x_detrend', 'z_detrend', 'y_detrend', 'PAT', 'PATmedian',\
       'PATdetrend', 'footpeak', 'footpeakmedian', 'HRmedian', 'HR', 'SpO2',\
       'RR', 'RRsqi', 'PPGamp', 'PPGampmedian']
features_and_label_format = {
        'label':[]
    }
STORAGE_PATH = "/home/heoyoun1/scratch/features3"
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

def identify_R_peaks(ecgValues,sampling_rate = SAMPLING_RATE):
  signals, info = nk.ecg_process(ecgValues,sampling_rate)

  # with open("info.txt",'w') as file:
  #    file.write(str(info))

  # with open('signals.txt','w') as file:
  #    file.write(str(signals))

  return info['ECG_R_Peaks']
     
def getAverageDistance_Between_R_peaks(r_peaks_indices,times):
  current = 0
  next = 1
  total_diff = 0
  while(next != len(r_peaks_indices)-1):
    with open('r_peaks.txt','a') as file:
       file.write(str(r_peaks_indices[current]) + '\n')
    diff = (times[r_peaks_indices[next]] - times[r_peaks_indices[current]])
    total_diff += diff
    current += 1
    next += 1
  with open('r_peaks.txt','a') as file:
     file.write(str(len(r_peaks_indices)-1))
  print(total_diff)
  return total_diff/(len(r_peaks_indices)-1)

def find_QRS_complex(ecg_Values,r_peaks_indices):
  qrs_complexes = []
  troughs, _ = find_peaks(-ecg_Values)
  
  r_peak_current = 0
  troughs_q = 0
  troughs_s = 1

  while( troughs[troughs_q] > r_peaks_indices[r_peak_current] and troughs[troughs_s] > r_peaks_indices[r_peak_current]):
    r_peak_current +=1

  while( troughs[troughs_q] < r_peaks_indices[r_peak_current] and troughs[troughs_s] < r_peaks_indices[r_peak_current]):
    troughs_q += 1
    troughs_s += 1

  while(troughs_s < len(troughs) and r_peak_current < len(r_peaks_indices)):
    current_r = r_peaks_indices[r_peak_current]
    before_trough = troughs[troughs_q]
    after_trough = troughs[troughs_s]

    if (before_trough < current_r and after_trough > current_r):
      qrs_complexes.append({
        'q':before_trough,
        'r':current_r,
        's':after_trough
      })
      r_peak_current += 1

    elif (current_r > before_trough and current_r > after_trough):
      troughs_q += 1
      troughs_s += 1

    else:
       r_peak_current += 1

  
  # with open('qrs.txt','w') as file:
  #    file.write(str(qrs_complexes))

  return qrs_complexes

def get_Average_Same_Type_Time_difference(times,qrs_complexes):
  total_p_diff = 0
  total_q_diff = 0
  total_r_diff = 0
  total_s_diff = 0
  total_t_diff = 0
  current= 0
  next = 1
  length = len(qrs_complexes)
  
  while(next < len(qrs_complexes)):
    current_complex = qrs_complexes[current]
    next_complex = qrs_complexes[next]
    # with open('complex.txt','a') as file:
    #    file.write(str(current_complex) + '\n')
    total_p_diff += (times[next_complex[0]]) - (times[current_complex[0]])
    total_q_diff += (times[next_complex[1]]) - times[current_complex[1]]
    total_r_diff += (times[next_complex[2]]) - times[current_complex[2]]
    total_s_diff += (times[next_complex[3]]) - times[current_complex[3]]
    total_t_diff += (times[next_complex[4]]) - times[current_complex[4]]
    current += 1
    next += 1
  # print(total_r_diff)
  return {
     'avg_p_diff':(total_p_diff/(length-1)),
     'avg_q_diff':(total_q_diff/(length-1)),
     'avg_r_diff':(total_r_diff/(length-1)),
     'avg_s_diff':(total_s_diff/(length-1)),
     'avg_t_diff':(total_t_diff/(length-1))
  }

def applyBandPassFilter(lowcut,highcut,signal,sampling_rate):
  sos = scipy.signal.butter(2, [lowcut, highcut], btype='band', fs=sampling_rate, output='sos')
  filtered_signal = scipy.signal.sosfilt(sos, signal)

  return filtered_signal


def getMinInRange(start,end,signal):
    min_index = start
    for i in range(start+1,end):
        if signal[i] < signal[min_index]:
            min_index = i

    return min_index

def getMaxInRange(start,end,signal):
    max_index = start
    for i in range(start+1,end):
        if signal[i] > signal[max_index]:
            max_index = i
    return max_index

def detect_PQRST_waves(r_peaks,signal):
    complexes = []
    signal_length = len(signal)
    for r in r_peaks:
        q_range = r-8
        if (q_range) < 0:
            continue
        s_range = r+8
        if (s_range) >= signal_length:
            continue

        q = getMinInRange(q_range,r,signal)
        s = getMinInRange(r,s_range,signal)

        p_range = q - 20
        t_range = s + 40

        if(p_range < 0):
            continue
        if (t_range >= signal_length):
           continue
        p = getMaxInRange(p_range,q,signal)
        t = getMaxInRange(s,t_range,signal)

        complexes.append([p,q,r,s,t])

    return complexes

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
    print(f"Currently Processing {name} at path: {path}...",flush = True)
    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    print(df['sleepstage'].unique(),flush = True)
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
    positive_sections = [(start,end+1) for start, end in zip(start_indices, end_indices)]
    print(f"Finished procesing {name}",flush = True)
    
    return {
        'name':name,
        'path':path,
        'sleep_periods':positive_sections
    }
def normalize(data):
    data_normalized = (data - data.min()) / (data.max() - data.min())
    return data_normalized
def roundToFirstdigit(num):
   first_2_digits = float(str(num)[0:2])
   returnVal = first_2_digits/10
   return round(returnVal)
def extractFeaturesHelper(df,window_time,window_move_time):
    '''
    label meanings:
    0 -> non-event
    1 -> including respiratory event
    2 -> non-event but rem_sleep
    '''
    features_and_label = {
        'label':[]
    }
    rem_sleep_label = 5
    features_and_label['average_distance_between_t'] = []
    features_and_label['average_distance_between_p'] = []
    features_and_label['average_distance_between_r'] = []
    features_and_label['average_distance_between_q'] = []
    features_and_label['average_distance_between_s'] = []
    start = 0
    end = SAMPLING_RATE * window_time
    increment = SAMPLING_RATE* window_move_time

    while(end < len(df)):
        current_sample = df.iloc[start:end,:]
        times = current_sample['time'].values
        resp_events = current_sample['resp_events'].values
        resp_events = np.array(list(map(roundToFirstdigit,resp_events)))
        sleep_stage = np.array(list(map(roundToFirstdigit,current_sample['sleepstage'].values)))
        any_positive = np.any(resp_events > 0)
    
        rem_sleep = np.any(sleep_stage == rem_sleep_label)
        label = 0
        if (any_positive):
            label = 1
        else:
           if rem_sleep:
              label = 2
        print(f'sleep stage values {np.unique(sleep_stage)} and label {label}',flush = True)
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
        try:
            ecgValues = current_sample['ecgProcessed'].values

            #apply a bandpass filter with typical parameters for a ecgSignal

            filtered = applyBandPassFilter(0.5,40,ecgValues,SAMPLING_RATE)

            # #detrend the data
            # detrended = detrend(filtered)

            # normalized_values_0_1 = (detrended_values - detrended_values.min())/(detrended_values.max() - detrended_values.min())

            # normalized_values_minus1_1 = normalized_values_0_1 * 2 -1

            signals,info = nk.ecg_process(filtered,SAMPLING_RATE)
            cleaned = signals['ECG_Clean']
            r_peaks = info['ECG_R_Peaks']
                
            pqrst_complexes = detect_PQRST_waves(r_peaks,cleaned)
            dists = get_Average_Same_Type_Time_difference(times,pqrst_complexes)
            # r_peaks = identify_R_peaks(ecgValues=ecgValues)
            # print('average distance between r')
            # getAverageDistance_Between_R_peaks(r_peaks_indices=r_peaks,times=times)
            # qrs_complexes = find_QRS_complex(ecgValues,r_peaks)
            # print('average distance between q and s')
            # same_type_avg_differences = get_Average_StoS_RtoR_QtoQ_difference(times,qrs_complexes)\
            features_and_label['average_distance_between_p'].append(dists['avg_p_diff'])
            features_and_label['average_distance_between_q'].append(dists['avg_q_diff'])
            features_and_label['average_distance_between_s'].append(dists['avg_s_diff'])
            features_and_label['average_distance_between_r'].append(dists['avg_r_diff'])
            features_and_label['average_distance_between_t'].append(dists['avg_t_diff'])
        except Exception as e :
            print(f"Could not extract qrs features due to : {e}",flush = True)
            features_and_label['average_distance_between_p'].append(None)
            features_and_label['average_distance_between_q'].append(None)
            features_and_label['average_distance_between_s'].append(None)
            features_and_label['average_distance_between_r'].append(None)
            features_and_label['average_distance_between_t'].append(None)
        start,end = start + increment,end+increment

    return pd.DataFrame(features_and_label)

def extract_features(name_and_sleep_periods,window_time,window_move_time):

    name = name_and_sleep_periods['name']
    path = name_and_sleep_periods['path']
    print(f"Started extracting features from {name}")
    sleep_periods = name_and_sleep_periods['sleep_periods']
    raw = mne.io.read_raw_edf(path)
    df = raw.to_data_frame()
    features_dfs = []
    for start,end in sleep_periods:
        sleep_period = df.iloc[start:end,:]
        features_df = extractFeaturesHelper(sleep_period,window_time,window_move_time)
        features_dfs.append(features_df)
    all_features_df = pd.DataFrame()
    try:
        print('attempign concat of dfs',flush = True)
        all_features_df = pd.concat(features_dfs)
    except Exception as e:
        print('error when attempting to concat dfs',flush = True)
    print(f"Finished extracting features from {name}")
    print(f"Started dumping {len(all_features_df)} rows into {name}.csv",flush = True)
    all_features_df.to_csv(f'{STORAGE_PATH}/{name}.csv',index = False)
    print(f"Finished dumping {len(all_features_df)} rows into {name}.csv",flush = True)
    return {
        'name':name,
        'features':all_features_df
    }

def equally_sample_rows(path):
    df = pd.read_csv(path)
    df = df.dropna()

    # Filter rows where label is 1
    df_label_1 = df[df['label'] == 1]

    # Count the number of rows with label 1
    n_label_1 = df_label_1.shape[0]

    # Filter rows where label is 0
    df_label_0 = df[df['label'] == 0]

    # Sample rows from the label 0 DataFrame
    # The `min()` function ensures you don't sample more rows than exist
    df_label_0_sampled = df_label_0.sample(n=min(n_label_1, len(df_label_0)), random_state=42)  # random_state for reproducibility

    # Concatenate the two DataFrames
    result_df = pd.concat([df_label_1, df_label_0_sampled]).reset_index(drop=True)

    # Display the result DataFrame
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to extract features from varying times of data')
    os.makedirs(STORAGE_PATH,exist_ok=True)
    parser.add_argument('window_time_in_seconds', type=int, help='A sample integer argument')
    parser.add_argument('time_movement_in_seconds',type=int, help = 'Time that window moves')
    # Parse the arguments
    args = parser.parse_args()
    window_time = args.window_time_in_seconds
    window_move_time = args.time_movement_in_seconds
    t0 = datetime.now()
    os.makedirs(STORAGE_PATH, exist_ok=True)
    print(f'removing sleep periods started {t0}',flush = True)
    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes=npus)
    paths_and_names = get_Paths_and_Names()
    print(f"There are {len(paths_and_names)} files to process")

    results = [pool.apply_async(extract_sleep_periods,args = (x,)) for x in paths_and_names]
    sleep_periods = [p.get() for p in results]
    pool.close()
    pool.join()

    t1 = datetime.now()

    print(f'removing sleep periods ended at {t1}',flush = True)
    print(f"Duration was {t1-t0}",flush = True)

    t0 = datetime.now()
    print(f"extracting features started {t0}",flush = True)
    npus = int(os.environ.get('SLURM_CPUS_PER_TASK',default = 4))
    pool = mp.Pool(processes=npus)
    paths_and_names = get_Paths_and_Names()

    results = [pool.apply_async(extract_features,args = (x,window_time,window_move_time)) for x in sleep_periods]
    features = [p.get() for p in results]
    pool.close()
    pool.join()

    t1 = datetime.now()
    print(f"extracting features ended {t1}",flush = True)
    print(f"Duration was {t1-t0}",flush = True)

