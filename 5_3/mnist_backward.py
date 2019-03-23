#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 20:18
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

# attempting to perform BLAS operation using StreamExecutor without BLAS support
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35

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

    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

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
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/5_3/mnist_backward.py
WARNING:tensorflow:From D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/5_3/mnist_backward.py:67: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please use urllib or similar directly.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting ./data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting ./data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting ./data/t10k-images-idx3-ubyte.gz
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Instructions for updating:
Extracting ./data/t10k-labels-idx1-ubyte.gz
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 20:26:49.861048: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 20:26:50.267392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 20:26:50.273313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 20:26:50.847509: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 20:26:50.851439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 20:26:50.853568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 20:26:50.856077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 716 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
After 1 training step(s), loss on training batch is 2.85387.
2019-03-23 20:26:51.194695: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 1001 training step(s), loss on training batch is 0.3081.
After 2001 training step(s), loss on training batch is 0.368691.
After 3001 training step(s), loss on training batch is 0.282282.
After 4001 training step(s), loss on training batch is 0.21218.
After 5001 training step(s), loss on training batch is 0.227694.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
After 6001 training step(s), loss on training batch is 0.209883.
After 7001 training step(s), loss on training batch is 0.19605.
After 8001 training step(s), loss on training batch is 0.195461.
After 9001 training step(s), loss on training batch is 0.199832.
After 10001 training step(s), loss on training batch is 0.182735.
After 11001 training step(s), loss on training batch is 0.172309.
After 12001 training step(s), loss on training batch is 0.169235.
After 13001 training step(s), loss on training batch is 0.167963.
After 14001 training step(s), loss on training batch is 0.160688.
After 15001 training step(s), loss on training batch is 0.161342.
After 16001 training step(s), loss on training batch is 0.16016.
After 17001 training step(s), loss on training batch is 0.151293.
After 18001 training step(s), loss on training batch is 0.150226.
After 19001 training step(s), loss on training batch is 0.140229.
After 20001 training step(s), loss on training batch is 0.142941.
After 21001 training step(s), loss on training batch is 0.15283.
After 22001 training step(s), loss on training batch is 0.146468.
After 23001 training step(s), loss on training batch is 0.155297.
After 24001 training step(s), loss on training batch is 0.142734.
After 25001 training step(s), loss on training batch is 0.136293.
After 26001 training step(s), loss on training batch is 0.142318.
After 27001 training step(s), loss on training batch is 0.13962.
After 28001 training step(s), loss on training batch is 0.154685.
After 29001 training step(s), loss on training batch is 0.131024.
After 30001 training step(s), loss on training batch is 0.13543.
After 31001 training step(s), loss on training batch is 0.131898.
After 32001 training step(s), loss on training batch is 0.130529.
After 33001 training step(s), loss on training batch is 0.134887.
After 34001 training step(s), loss on training batch is 0.132173.
After 35001 training step(s), loss on training batch is 0.130442.
After 36001 training step(s), loss on training batch is 0.131539.
After 37001 training step(s), loss on training batch is 0.143655.
After 38001 training step(s), loss on training batch is 0.130881.
After 39001 training step(s), loss on training batch is 0.136281.
After 40001 training step(s), loss on training batch is 0.132111.
After 41001 training step(s), loss on training batch is 0.137283.
After 42001 training step(s), loss on training batch is 0.12635.
After 43001 training step(s), loss on training batch is 0.13205.
After 44001 training step(s), loss on training batch is 0.128063.
After 45001 training step(s), loss on training batch is 0.126283.
After 46001 training step(s), loss on training batch is 0.126209.
After 47001 training step(s), loss on training batch is 0.128365.
After 48001 training step(s), loss on training batch is 0.128746.
After 49001 training step(s), loss on training batch is 0.123466.

Process finished with exit code 0
'''
