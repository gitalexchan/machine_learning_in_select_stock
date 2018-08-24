"""
This file is used to transform the sql data into numpy data
which is used to load in tensorflow for analysis

train_x: get the train data
train_y: get the train label data
test_x: get the test data
test_y: get the test label data
"""


import MySQLdb
import numpy as np


#get data from sql databases
class database():
    def __init__(self):
        self.conn = MySQLdb.connect(
            host='127.0.0.1',
            user='root',
            passwd='sangomine',
            port=3306,
            db='original_data',
            charset = 'utf8'
        )


    def train_x(self):
        with self.conn:
            cur = self.conn.cursor()
            sql = 'SELECT roe_yearly,gross_rate_yearly,targ_yearly,epsg_yearly,' \
                  'TTM_yearly,eps_yearly,er_yearly FROM train_x'
            cur.execute(sql)
            data = cur.fetchall()
            data_arr = np.array(list(data)).reshape(-1,7)
            cur.close()
            return data_arr


    def train_y(self):
        with self.conn:
            cur = self.conn.cursor()
            sql = 'SELECT top_one_quarter,tail_three_quarter FROM train_y'
            cur.execute(sql)
            data = cur.fetchall()
            data_arr = np.array(list(data)).reshape(-1,2)
            cur.close()
            return data_arr


    def test_x(self):
        with self.conn:
            cur = self.conn.cursor()
            sql = 'SELECT roe_yearly,gross_rate_yearly,targ_yearly,eps_yearly,epsg,' \
                  'TTM_yearly,er_yearly FROM test_x'
            cur.execute(sql)
            data = cur.fetchall()
            data_arr = np.array(list(data)).reshape(-1,7)
            cur.close()
            return data_arr


    def test_y(self):
        with self.conn:
            cur = self.conn.cursor()
            sql = 'SELECT top_one_quarter,tail_three_quarter FROM test_y'
            cur.execute(sql)
            data = cur.fetchall()
            data_arr = np.array(list(data)).reshape(-1,2)
            cur.close()
            return data_arr

if __name__ == '__main__':
    db = database()
    train_x = db.train_x()
    print(train_x.shape)
    train_y = db.train_y()
    print(train_y.shape)
    test_x = db.test_x()
    print(test_x.shape)
    test_y = db.test_y()
    print(test_y.shape)
