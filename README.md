# machine_learning_in_select_stock
## 第一份工作，给老板写一个选股的模型


用到的工具是tushare。
这个工具有点麻烦，在安装的时候需要失败后安装它提示的一些package。并不是因为pip的原因。tushare安装过程中会提示你需要安装的包名称，按照它提示的去安装完所有的支持包，就能正常使用。

## 第一份计划：已经实现，初步准确度为百分之40多，需要改进。（神经网络层数）
### 设计方案 
通过对每只股票基本面数据的规整，得到影响企业股票价格的企业各方面数据，把企业的运营数据和生产数据作为深度学习的特征进行整合和清洗；获得每只股票的股票数据，设定在该年度持有该股票期限为半年，随机抽取100天的半年的收益率并计算总收益率的平均值。将所有股票收益率分类并在基本面数据后进行标注，获得每只股票的基本面数据和收益率分类的总数据。对该数据进行分层抽样分离数据集，对测试集进行深度学习，然后获得的模型用验证集学习。对比验证集学习后的标签和原标签对比，获得正确率，评估模型效果。
### 步骤
#### tushare安装
#### 获得2016年全年的每只股票基本面数据
#### 持有期半年股票数据收益率计算并分类（4类）
##### 获得2015/06/01 - 2016/12/31的每只股票日线前复权股市数据
##### add a column named['profit_half_year_ago'] at the end of the dataframe, data comes from the 2015/06/01 of '开盘价'
##### add a column named['profit_rate'] at the end of the dataframe
##### df['profit_rate'] = (df['收盘价'] - df['profit_half_year_ago'])/df['profit_half_year_ago'])
##### delete all columns except df['profit_rate']
##### remove 2015/06/01 - 2015/12/31 data, random select 100 days and calculate the mean values.
##### add the mean values named ['profit_rate'] to the end of related code
     
#### deeplearning of datasets
   Code in the fold for review.
     

