PATH = "/scratch/alim/overnight_validation/MUSE-PSG/"
import os

patients = {

}

def read_files_in_directory(directory_path):
    try:
        #print("Finding all MUSE-PSG files")
        # Get a list of all files in the directory
        file_list = sorted([entry.name for entry in os.scandir(directory_path) if
                    entry.is_file() and any(keyword in entry.name for keyword in ['ppg', 'events'])])


                     
        # Read the contents of each file
        #print("Findind all file names completed")

        return file_list
    except Exception as e:
        print(f"Error reading files: {e}")



if __name__ == "__main__":

    file_list = read_files_in_directory(PATH)

    for file_name in file_list:
        parts = file_name.rsplit('_',1)
        patient_name = parts[0]
        
        if patient_name not in patients:
            patients[patient_name] = {
                'ppg':'',
                'events':''
            }

        if 'ppg' in file_name:
            patients[patient_name]['ppg'] = PATH + file_name

        if 'events' in file_name:
            patients[patient_name]['events'] = PATH + file_name

    print (patients)
