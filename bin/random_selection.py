import os
import pandas as pd
import numpy as np




class GetComnTim():

    dir_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(dir_path, 'stock_k_data')
    save_path = os.path.join(dir_path,'stock_sum')

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
            # print(type(code_name))

            with open(path_name, 'r') as f:
                with open(os.path.join(GetComnTim.save_path,'2016_sum.csv'), 'a') as sf:
                    df = pd.read_csv(f)
                    if df.shape[0] < 400:
                        continue
                    else:
                        df = df.drop([df.columns[0],'code'], axis = 1).set_index(keys = 'date')
                        df.index = pd.to_datetime(df.index)
                        df['profit'] = (df['close'] - df['open'].shift(120))/df['open'].shift(120)
                        df_2016 = df['2016'].sample(frac = 0.33)
                        # *avg is float type data.
                        df_2016_avg = df_2016['profit'].mean()
                        # code name is str type
                        avg_list2016 = ','.join([code_name,str(df_2016_avg)])
                        sf.write(avg_list2016 + '\n')

            with open(path_name, 'r') as f:
                with open(os.path.join(GetComnTim.save_path,'2017_sum.csv'), 'a') as gf:
                    df = pd.read_csv(f)
                    df = df.drop([df.columns[0],'code'], axis = 1).set_index(keys = 'date')
                    df.index = pd.to_datetime(df.index)
                    df['profit'] = (df['close'] - df['open'].shift(120))/df['open'].shift(120)
                    df_2017 = df['2017'].sample(frac = 0.33)
                    # *avg is float type data.
                    df_2017_avg = df_2017['profit'].mean()
                    # code name is str type
                    avg_list2017 = ','.join([code_name,str(df_2017_avg)])
                    gf.write(avg_list2017 + '\n')







if __name__ == '__main__':
    a = GetComnTim()
    a.csv_path()


