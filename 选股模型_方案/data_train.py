import os
import tensorflow as tf
import data_inference
from data_from_sql import database


# basic para
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.75
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000

MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "code_selection.ckpt"


# train step
def train(train_x_matrix, train_y_matrix):
    x = tf.placeholder(tf.float32, [None, data_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, data_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = data_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               train_x_matrix.shape[0] / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            # # use batch to speed up the process
            # start = (i * BATCH_SIZE) // train_x_matrix.shape[0]
            # end = min(start + BATCH_SIZE, train_x_matrix.shape[0])
            # xs = train_x_matrix[start: end]
            # ys = train_y_matrix[start: end]

            # _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: train_x_matrix, y_: train_y_matrix})

            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is %g." % (step, loss_value))
                saver.save(sess ,os.path.join( MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)


def main (argv=None):
    db = database()
    train_x, train_y = db.train_x(), db.train_y()
    print(train_y)
    print(train_x.shape[0])
    train(train_x, train_y)


if __name__ == '__main__':
    main()









