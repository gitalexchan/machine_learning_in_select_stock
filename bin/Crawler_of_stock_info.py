
import os
import tushare as ts
import pandas as pd



# time_info =
#基本面数据，需要有

#get the path of needed filename or dirname
LastFilePath = os.path.dirname(os.path.dirname(__file__))
data_file_path = os.path.join(LastFilePath, 'stock_k_data')


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
        #stock2year1 is tushare:get_k_data;

        stock2year1 = ts.get_k_data(stock_code, '2016-01-01', '2018-01-01')
        csv_name = os.path.join(data_file_path, "{}.csv".format(stock_code))
        stock2year1.to_csv(csv_name)
        return csv_name


def store_fund_data(quarter_list):
    stock2year_path = os.path.join(LastFilePath, "stock_fundm_info")
    for fun_year,fun_quarter in quarter_list:
        
        #every dataframe you craw down all needs remove the duplicated row. Only need keep the first row of duplicates.
        
        # stock2year_report is tushare:get_report_data  (fundamental data).
        stock2year_report = ts.get_report_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        # stock2year_prof is tushare.get_profit_data  (fundamental data).
        stock2year_prof = ts.get_profit_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        # stock2year_opera is tushare.get_operation_data (fundamental data).
        stock2year_opera = ts.get_operation_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        #stock2year_grow is tushare.get_growth_data (fundamental data).
        stock2year_grow = ts.get_growth_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        #stock2year_debt is tushare.get_debtpaying_data (fundamental data).
        stock2year_debt = ts.get_debtpaying_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        #stock2year_cash is tushare.get_cashflow_data (fundamental data).
        stock2year_cash = ts.get_cashflow_data(fun_year,fun_quarter).drop_duplicates(keep='first')
        #stock2year_comb is to combine all the stock2year data of same year and quarter in a same stock code.
        stock2year_list = [stock2year_report,stock2year_prof,stock2year_opera,stock2year_grow, \
                           stock2year_debt,stock2year_cash]
        for every_fund_element in stock2year_list:
            every_fund_element = every_fund_element.set_index('code')
        #use pandas concat to combine all the dataframe along columns.
        total_fund = pd.concat(stock2year_list,axis=1)
        HeadName = fun_year + "/" + fun_quarter + "_" + "fundamt_info"
        CsvName = os.path.join(stock2year_path, "{}.csv".format(HeadName))
        total_fund.to_csv(CsvName)



# def random_select(filename):


if __name__ == '__main__':
    # get the directory of output file path: data_file_path


    year,quarter = 2016,3
    list = get_stock_code(year,quarter)
    store_stock_data(list)
    fund_list = [["2016","1"],["2016","2"],["2016",3],["2016","4"],\
                 ["2017","1"],["2017","2"],["2017",3],["2017","4"]]
