import mne
import pandas as pd
import scipy
from scipy.signal import detrend
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os
ANNE_SAMPLING_RATE = 100
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

def plot_PQRST(times,pqrst_waves,signal,index):
  plt.clf()
  plt.figure(figsize = (20,5))

  plt.plot(times,signal,color = 'green')
  for wave in pqrst_waves:
    plt.scatter(x = times[wave[0]],y=signal[wave[0]],s = 100, color = 'red')
    plt.scatter(x = times[wave[1]],y = signal[wave[1]],s = 100, color = 'purple')
    plt.scatter(x = times[wave[2]],y = signal[wave[2]],s = 100, color = 'black')
    plt.scatter(x = times[wave[3]],y = signal[wave[3]],s = 100, color = 'blue')
    plt.scatter(x = times[wave[4]],y = signal[wave[4]],s = 100, color = 'pink')

  plt.savefig(f"./plots1/{index}.png")
  plt.clf()

def applyBandPassFilter(lowcut,highcut,signal,sampling_rate):
  sos = scipy.signal.butter(2, [lowcut, highcut], btype='band', fs=sampling_rate, output='sos')
  filtered_signal = scipy.signal.sosfilt(sos, signal)

  return filtered_signal
if __name__ == '__main__':

    path = "/scratch/alim/overnight_validation/ANNE-PSG231215/20-08-20-21_33_30.C1442.L1215.185/20-08-20-21_33_30.C1442.L1215.185-features.edf"

    raw = mne.io.read_raw_edf(path)

    df = raw.to_data_frame()

    separated = separate_Data_by_Respiratory_Event(df,3)
    index = 0
    for label in separated:
        
        for data in separated[label]:
            ecgValues = data['ecgProcessed'].values[0:min(500,len(data['ecgProcessed'].values))]
            times = data['time'].values[0:min(500,len(data['ecgProcessed'].values))]
            
            filtered = applyBandPassFilter(0.5,40,ecgValues,ANNE_SAMPLING_RATE)

            signals, info = nk.ecg_process(filtered,ANNE_SAMPLING_RATE)
            cleaned = signals['ECG_Clean'].values
            detrended_values = detrend(cleaned)
            normalized_values_0_1 = (detrended_values - detrended_values.min())/(detrended_values.max() - detrended_values.min())

            normalized_values_minus1_1 = normalized_values_0_1 * 2 -1

            # print(len(signals),len(normalized_values_minus1_1))
            # they are the same length ^
            r_peaks = info['ECG_R_Peaks']
            
            pqrst_waves = detect_PQRST_waves(r_peaks,normalized_values_minus1_1)
            # with open('pqrst.txt','a') as file:
            #    file.write(str(pqrst_waves))

            plot_PQRST(times,pqrst_waves,normalized_values_minus1_1,index)

            index+= 1