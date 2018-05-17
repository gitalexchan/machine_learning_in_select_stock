
import os
import tushare as ts
import pandas as pd



# time_info =
#基本面数据，需要有


def get_stock_code(selec_year,selec_quarter): #time is a list
    stock_info = ts.get_report_data(selec_year,selec_quarter)

#First obtain the codes of every company

    stock_code = stock_info["code"]
    stock_list = []
    for i in stock_code:
        stock_list.append(i)
    return stock_list

def store_stock_data(list_stock):
    for stock_code in list_stock:
        #stock2year1 is get_k_data;
        #stock2year2 is
        stock2year1 = ts.get_k_data(stock_code, '2016-01-01', '2018-01-01')
        csv_name = os.path.join(data_file_path, "{}.csv".format(stock_code))
        stock2year1.to_csv(csv_name)


# def random_select(filename):


if __name__ == '__main__':
    # get the directory of output file path: data_file_path
    last_file_path = os.path.dirname(os.path.dirname(__file__))
    data_file_path = os.path.join(last_file_path, 'stock_k_data')
    year,quarter = 2016,3
    list = get_stock_code(year,quarter)
    store_stock_data(list)









