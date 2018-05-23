import os
import pandas as pd
import numpy as np




class GetComnTim():

    dir_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(dir_path, 'stock_k_data')
    save_path = os.path.join(dir_path,'stock_k_sum.csv')

    def __init__(self):
        pass

    def csv_path(self):

        global com_time
        csv_path_list = []
        csv_dir = os.listdir(GetComnTim.file_path)
        for csv_name in csv_dir:
            csv_path = os.path.join(GetComnTim.file_path, csv_name)
            if os.path.isfile(csv_path):
                if os.path.splitext(csv_path)[1] == '.csv':
                    csv_path_list.append(csv_path)
        print(csv_path_list)



        for path_name in csv_path_list:

            code_name = os.path.split(path_name)[1][0:6]
            # print(code_name)
            com_time = []
            with open(path_name, 'r') as f:
                with open(GetComnTim.save_path, 'a') as sf:
                    df = pd.read_csv(f)
                    df.rename(columns ={'date':code_name}, inplace= True)
                    time_series = df.iloc[:,1]
                    #the max length of date is 488, 488 - 180 = 308
                    #drop the code if the days less than 300 which means that this code has less information
                    #or suspension over 6 months.
                    if len(time_series) > 300:
                        time_array = np.array(time_series)
                        time_list = time_array.tolist()
                        time_list.insert(0,code_name)
                        time_line = ','.join(time_list)
                        sf.write(time_line+'\n')



if __name__ == '__main__':
    a = GetComnTim()
    a.csv_path()

