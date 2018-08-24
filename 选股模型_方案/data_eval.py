"""
This file is used to test the code into deep-learning model
It will give the classes of the code and probability that code in classes
The model comes from the data_train.py

"""
import numpy as np
import tensorflow as tf
import data_inference
import data_train
import data_from_sql


def evaluate(test_x, test_y):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, test_x.shape[1]], name='x-input'
        )
        y_ = tf.placeholder(
            tf.float32, [None, 2], name='y-input'
        )
        validate_feed = {
            x:test_x,
            y_:test_y
        }

        y = data_inference.inference(x, None)

        pre_result = tf.nn.relu(tf.clip_by_value(y,1e-10,1.0))
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(data_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                data_train.MODEL_SAVE_PATH
            )
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_list = sess.run(pre_result,feed_dict=validate_feed)
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step, validation accuracy = %g" % (global_step, accuracy_score))
                np.set_printoptions(suppress=True)
                print(accuracy_list)
                # restore the final list in pandas dataframe
                np.savetxt("accuracy_list.csv",accuracy_list,delimiter=',',fmt="%.2f")

            else:
                print("Checkpoints file is not existing")
                return


def main(argv=None):
    db = data_from_sql.database()
    test_x, test_y = db.test_x(), db.test_y()
    train_x = db.train_x()
    total_x = np.concatenate((train_x,test_x),axis = 0)
    print(total_x.shape[0])
    evaluate(test_x, test_y)


if __name__ == '__main__':
    main()

