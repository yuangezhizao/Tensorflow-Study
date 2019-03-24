#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/24 0024 10:08
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import os

import mnist_forward
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/6_1/mnist_backward.py
WARNING:tensorflow:From D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/6_1/mnist_backward.py:67: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please use urllib or similar directly.
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Extracting ./data/train-images-idx3-ubyte.gz
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting ./data/train-labels-idx1-ubyte.gz
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting ./data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Extracting ./data/t10k-labels-idx1-ubyte.gz
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-24 10:19:25.381775: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-24 10:19:26.249987: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-24 10:19:26.257282: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:19:27.863275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:19:27.866536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:19:27.868701: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:19:27.874550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-24 10:19:28.378162: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 1 training step(s), loss on training batch is 2.87809.
After 1001 training step(s), loss on training batch is 0.342129.
After 2001 training step(s), loss on training batch is 0.304886.
After 3001 training step(s), loss on training batch is 0.277106.
After 4001 training step(s), loss on training batch is 0.242945.
After 5001 training step(s), loss on training batch is 0.243571.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
After 6001 training step(s), loss on training batch is 0.187642.
After 7001 training step(s), loss on training batch is 0.187937.
After 8001 training step(s), loss on training batch is 0.227444.
After 9001 training step(s), loss on training batch is 0.175261.
After 10001 training step(s), loss on training batch is 0.175079.
After 11001 training step(s), loss on training batch is 0.182911.
After 12001 training step(s), loss on training batch is 0.163193.
After 13001 training step(s), loss on training batch is 0.159741.
After 14001 training step(s), loss on training batch is 0.159472.
After 15001 training step(s), loss on training batch is 0.157042.
After 16001 training step(s), loss on training batch is 0.168561.
After 17001 training step(s), loss on training batch is 0.153753.
After 18001 training step(s), loss on training batch is 0.162056.
After 19001 training step(s), loss on training batch is 0.152495.
After 20001 training step(s), loss on training batch is 0.147137.
After 21001 training step(s), loss on training batch is 0.15152.
After 22001 training step(s), loss on training batch is 0.149748.
After 23001 training step(s), loss on training batch is 0.143643.
After 24001 training step(s), loss on training batch is 0.147914.
After 25001 training step(s), loss on training batch is 0.145608.
After 26001 training step(s), loss on training batch is 0.143308.
After 27001 training step(s), loss on training batch is 0.139329.
After 28001 training step(s), loss on training batch is 0.146811.
After 29001 training step(s), loss on training batch is 0.139699.
After 30001 training step(s), loss on training batch is 0.140325.
After 31001 training step(s), loss on training batch is 0.131904.
After 32001 training step(s), loss on training batch is 0.136961.
After 33001 training step(s), loss on training batch is 0.130909.
After 34001 training step(s), loss on training batch is 0.135038.
After 35001 training step(s), loss on training batch is 0.127844.
After 36001 training step(s), loss on training batch is 0.161377.
After 37001 training step(s), loss on training batch is 0.127475.
After 38001 training step(s), loss on training batch is 0.128104.
After 39001 training step(s), loss on training batch is 0.133309.
After 40001 training step(s), loss on training batch is 0.129832.
After 41001 training step(s), loss on training batch is 0.128946.
After 42001 training step(s), loss on training batch is 0.130148.
After 43001 training step(s), loss on training batch is 0.134516.
After 44001 training step(s), loss on training batch is 0.123114.
After 45001 training step(s), loss on training batch is 0.124095.
After 46001 training step(s), loss on training batch is 0.125917.
After 47001 training step(s), loss on training batch is 0.130993.
After 48001 training step(s), loss on training batch is 0.138443.
After 49001 training step(s), loss on training batch is 0.126745.

Process finished with exit code 0
'''
