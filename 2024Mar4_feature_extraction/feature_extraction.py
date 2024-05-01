import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path 
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import detrend
import neurokit2 as nk
from scipy.signal import find_peaks
PLOTS_PATH = './PLOTS4'
AMPLITUDE = 1000
ANNE_SAMPLING_RATE = 100

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

def identify_R_peaks(ecgValues,sampling_rate = ANNE_SAMPLING_RATE):
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

    
   
def extractFeaturesHelper(separated,name):
    dict = {
       'label':[],
       'name':[]
    }
    dict['average_distance_between_t'] = []
    dict['average_distance_between_p'] = []
    dict['average_distance_between_r'] = []
    dict['average_distance_between_q'] = []
    dict['average_distance_between_s'] = []
    separated = separated.copy()

    for label in separated:
      label_data = separated[label]

      for data in label_data:
        times = data['time'].values
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
          
        #process the ecgProcessed
        # print('line 284')
        ecgValues = data['ecgProcessed'].values

        #apply a bandpass filter with typical parameters for a ecgSignal

        filtered = applyBandPassFilter(0.5,40,ecgValues,ANNE_SAMPLING_RATE)

        #detrend the data
        # detrended = detrend(filtered)

        # normalized_values_0_1 = (detrended_values - detrended_values.min())/(detrended_values.max() - detrended_values.min())

        # normalized_values_minus1_1 = normalized_values_0_1 * 2 -1

        signals,info = nk.ecg_process(filtered,ANNE_SAMPLING_RATE)
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
        dict['average_distance_between_p'].append(dists['avg_p_diff'])
        dict['average_distance_between_q'].append(dists['avg_q_diff'])
        dict['average_distance_between_s'].append(dists['avg_s_diff'])
        dict['average_distance_between_r'].append(dists['avg_r_diff'])
        dict['average_distance_between_t'].append(dists['avg_t_diff'])
      
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

    file_path = "./features.csv"
    try:
        # Attempt to write the DataFrame to a CSV file
        final_df.to_csv(file_path)
        print(f"DataFrame was successfully saved to {file_path}")
    except Exception as e:
        # Handle the exception if an error occurs
        print(f"An error occurred while trying to write to {file_path}: {e}")
    end = datetime.now()
    print(f"Job ended at {end}")
    duration = end - start
    print(f"Job took {duration} long")