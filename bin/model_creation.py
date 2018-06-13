import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('final_version_2016.csv')
bins = [-0.4,-0.056,-0.02,0.117,1.97]
data["y_label"] = pd.cut(data['profit'],bins,labels=['4','3','2','1'])
data = data.drop('profit',axis = 1)
data.set_index('code',inplace=True)

y_label_num = ['1','2','3','4']
#创建一个新dataframe，将每一类分层抽样的加进去
data_columns = data.columns.values.tolist()
test_df = pd.DataFrame(columns=data_columns)
for label in y_label_num:
    random_sample = data[data['y_label'] == label].sample(frac=0.25)
    print(random_sample.shape[0])
    data = data.drop(random_sample.index.tolist())
    test_df = pd.concat([test_df,random_sample],axis=0) 
    
    
train_y_ori = train_df['y_label']
train_x_ori = train_df.iloc[:,:-1]
test_y_ori = test_df['y_label']
test_x_ori = test_df.iloc[:,:-1]
###(1211,)
(1211, 19)
(404,)
(404, 19)###

train_y_ori = pd.get_dummies(train_y_ori).loc[:,y_label_num]
test_y_ori = pd.get_dummies(test_y_ori).loc[:,y_label_num]

train_y_matrix = train_y_ori.as_matrix().astype(np.float32)
train_x_matrix = train_x_ori.as_matrix().astype(np.float32)
test_y_matrix = test_y_ori.as_matrix().astype(np.float32)
test_x_matrix = test_x_ori.as_matrix().astype(np.float32)

#定义基本参数
INPUT_NODE = 19
OUTPUT_NODE = 4

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


#训练模型函数
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        
def train():
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion
    
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_x_matrix.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: train_x_matrix, y_: train_y_matrix}
        test_feed = {x: test_x_matrix, y_: test_y_matrix} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            start = (i * BATCH_SIZE)//train_x_matrix.shape[0]
            end = min(start + BATCH_SIZE, train_x_matrix.shape[0])
            xs = train_x_matrix[start:end]
            ys = train_y_matrix[start:end]
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))
        
def main(argv=None):
    train()

if __name__=='__main__':
    main()
    
    
###
After 0 training step(s), validation accuracy using average model is 0.277457 
After 1000 training step(s), validation accuracy using average model is 0.351775 
After 2000 training step(s), validation accuracy using average model is 0.361685 
After 3000 training step(s), validation accuracy using average model is 0.364162 
After 4000 training step(s), validation accuracy using average model is 0.360859 
After 5000 training step(s), validation accuracy using average model is 0.367465 
After 6000 training step(s), validation accuracy using average model is 0.36251 
After 7000 training step(s), validation accuracy using average model is 0.339389 
After 8000 training step(s), validation accuracy using average model is 0.35673 
After 9000 training step(s), validation accuracy using average model is 0.376548 
After 10000 training step(s), validation accuracy using average model is 0.402147 
After 11000 training step(s), validation accuracy using average model is 0.409579 
After 12000 training step(s), validation accuracy using average model is 0.40545 
After 13000 training step(s), validation accuracy using average model is 0.407102 
After 14000 training step(s), validation accuracy using average model is 0.406276 
After 15000 training step(s), validation accuracy using average model is 0.409579 
After 16000 training step(s), validation accuracy using average model is 0.409579 
###
