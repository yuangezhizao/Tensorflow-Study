#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 11:59
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 酸奶成本 9 元，酸奶利润 1 元
# 预测多了损失大，故不要预测多，故生成的模型会少预测一些
# 0导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 1定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2定义损失函数及反向传播方法
# 重新定义损失函数，使得预测多了的损失大，于是模型应该偏向少的方向预测
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            print('After %d training steps, w1 is: ' % (i))
            print(sess.run(w1), '\n')
    print('Final w1 is: \n', sess.run(w1))

'''C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_1/3.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 12:01:32.747612: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 12:01:33.493285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 12:01:33.499026: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 12:01:34.032768: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 12:01:34.036209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 12:01:34.038323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 12:01:34.041109: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 12:01:34.242621: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 0 training steps, w1 is: 
[[-0.80594873]
 [ 1.4873729 ]] 

After 500 training steps, w1 is: 
[[0.8732146]
 [1.006204 ]] 

After 1000 training steps, w1 is: 
[[0.9658064 ]
 [0.96982086]] 

After 1500 training steps, w1 is: 
[[0.9645447]
 [0.9682947]] 

After 2000 training steps, w1 is: 
[[0.9602475]
 [0.9742085]] 

After 2500 training steps, w1 is: 
[[0.96100295]
 [0.9699342 ]] 

Final w1 is: 
 [[0.9600407]
 [0.9733418]]

Process finished with exit code 0
'''
