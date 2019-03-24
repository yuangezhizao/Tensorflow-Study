#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/24 0024 11:19
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import os

import mnist_forward
import mnist_generateds  # 1
import tensorflow as tf

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'
train_num_examples = 60000  # 2


def backward():
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
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)  # 3

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()  # 4
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5

        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])  # 6
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()  # 7
        coord.join(threads)  # 8


def main():
    backward()  # 9


if __name__ == '__main__':
    main()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/6_2/mnist_backward.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From D:\yuangezhizao\Documents\PycharmProjects\Tensorflow-Study\6_2\mnist_generateds.py:62: string_input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(string_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\input.py:278: input_producer (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\input.py:190: limit_epochs (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\input.py:199: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\input.py:199: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\input.py:202: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From D:\yuangezhizao\Documents\PycharmProjects\Tensorflow-Study\6_2\mnist_generateds.py:63: TFRecordReader.__init__ (from tensorflow.python.ops.io_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TFRecordDataset`.
WARNING:tensorflow:From D:\yuangezhizao\Documents\PycharmProjects\Tensorflow-Study\6_2\mnist_generateds.py:87: shuffle_batch (from tensorflow.python.training.input) is deprecated and will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.shuffle(min_after_dequeue).batch(batch_size)`.
2019-03-24 11:36:34.346111: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-24 11:36:35.103492: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-24 11:36:35.109244: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 11:36:35.669899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 11:36:35.673115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 11:36:35.675092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 11:36:35.677888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
WARNING:tensorflow:From D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/6_2/mnist_backward.py:62: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
2019-03-24 11:36:36.371376: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 1 training step(s), loss on training batch is 2.24236.
After 1001 training step(s), loss on training batch is 0.220115.
After 2001 training step(s), loss on training batch is 0.152076.
After 3001 training step(s), loss on training batch is 0.205216.
After 4001 training step(s), loss on training batch is 0.167567.
After 5001 training step(s), loss on training batch is 0.158635.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
After 6001 training step(s), loss on training batch is 0.210692.
After 7001 training step(s), loss on training batch is 0.171899.
After 8001 training step(s), loss on training batch is 0.158845.
After 9001 training step(s), loss on training batch is 0.196471.
After 10001 training step(s), loss on training batch is 0.180448.
After 11001 training step(s), loss on training batch is 0.14437.
After 12001 training step(s), loss on training batch is 0.178022.
After 13001 training step(s), loss on training batch is 0.185863.
After 14001 training step(s), loss on training batch is 0.138782.
After 15001 training step(s), loss on training batch is 0.191732.
After 16001 training step(s), loss on training batch is 0.168763.
After 17001 training step(s), loss on training batch is 0.141043.
After 18001 training step(s), loss on training batch is 0.159804.
After 19001 training step(s), loss on training batch is 0.149362.
After 20001 training step(s), loss on training batch is 0.130862.
After 21001 training step(s), loss on training batch is 0.160967.
After 22001 training step(s), loss on training batch is 0.159692.
After 23001 training step(s), loss on training batch is 0.139557.
After 24001 training step(s), loss on training batch is 0.153192.
After 25001 training step(s), loss on training batch is 0.159082.
After 26001 training step(s), loss on training batch is 0.135244.
After 27001 training step(s), loss on training batch is 0.165513.
After 28001 training step(s), loss on training batch is 0.162942.
After 29001 training step(s), loss on training batch is 0.142064.
After 30001 training step(s), loss on training batch is 0.155545.
After 31001 training step(s), loss on training batch is 0.13796.
After 32001 training step(s), loss on training batch is 0.125994.
After 33001 training step(s), loss on training batch is 0.140172.
After 34001 training step(s), loss on training batch is 0.143543.
After 35001 training step(s), loss on training batch is 0.122421.
After 36001 training step(s), loss on training batch is 0.143754.
After 37001 training step(s), loss on training batch is 0.173271.
After 38001 training step(s), loss on training batch is 0.133238.
After 39001 training step(s), loss on training batch is 0.130665.
After 40001 training step(s), loss on training batch is 0.134306.
After 41001 training step(s), loss on training batch is 0.135702.
After 42001 training step(s), loss on training batch is 0.126434.
After 43001 training step(s), loss on training batch is 0.155796.
After 44001 training step(s), loss on training batch is 0.124808.
After 45001 training step(s), loss on training batch is 0.131072.
After 46001 training step(s), loss on training batch is 0.135592.
After 47001 training step(s), loss on training batch is 0.117375.
After 48001 training step(s), loss on training batch is 0.120023.
After 49001 training step(s), loss on training batch is 0.15125.

Process finished with exit code 0
'''
