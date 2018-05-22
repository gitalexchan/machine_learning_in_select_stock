import os
from itertools import islice



class GetComnTim():

    dir_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(dir_path, 'stock_k_data')

    def __init__(self):
        pass

    def csv_path(self):

        csv_path_list = []
        csv_name = os.listdir(GetComnTim.file_path)
        for name in csv_name:
            csv_path = os.path.join(GetComnTim.file_path, name)
            if os.path.isfile(csv_path):
                if os.path.splitext(csv_path)[1] == '.csv':
                    csv_path_list.append(csv_path)


        for path_name in csv_path_list:
            code_name = os.path.split(path_name)[1][0:7]
            com_time = []
            with open(path_name,'r') as f:
                farray = f.readlines()
                next(farray,None)
                for line in farray:
                    line = line.strip()
                    line = line.split(',')
                    date = line[1]
                    com_time.append('code_name'+','+date + '/n')
