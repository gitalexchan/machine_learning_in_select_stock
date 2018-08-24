predict_profit.csv：1442支股票的预测情况
股票数据维度：用来预测模型的数据特征向量
选股模型：股票预测的设计文档
data_from_sql.py：从sql数据库提取训练数据，训练数据标签，测试数据，测试数据标签
data_inference.py：神经网络前向传播算法
data_train.py：用神经网络对训练数据和训练标签进行训练，输出模型并存储。
data_eval.py：用神经网络训练好的模型进行预测