
import os
import tushare as ts



def create_k_dir(dir_name):
    last_path = os.path.dirname(os.path.dirname(__file__))
    print(last_path)
    dir_path = os.path.join(last_path,dir_name)
    print(dir_path)
    if os.path.isdir(dir_path):
        print(dir_name + ' is already existed')
        pass
    else:
        print('dir not exist')
        os.mkdir(dir_path)
    return dir_path


def get_codes(time_list):
    year,quarter = time_list
    stock_info = ts.get_report_data(year,quarter)
    codes = stock_info["code"]
    stock_list = []
    for code in codes:
        stock_list.append(code)
    print(stock_list)
    return stock_list


def download_k_data(code_list,time_series,dir):
    for code in code_list:
        start_day,end_day = time_series
        k_data = ts.get_k_data(code,start_day, end_day)
        k_file_path = os.path.join(dir,"{}.csv".format(code))
        k_data.to_csv(k_file_path)


if __name__ == '__main__':

    time_list = [2018,1]
    dir_name = create_k_dir('k_data')
    code_lists = get_codes(time_list)
    time_range = ['2017-06-01','2018-06-01']
    download_k_data(code_lists,time_range,dir_name)
