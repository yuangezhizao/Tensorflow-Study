#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 12:06
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 设损失函数 loss=(w+1)^2, 令 w 初值是常数 5。反向传播就是求最优 w，即求最小 loss 对应的 w 值
import tensorflow as tf

# 定义待优化参数 w 初值赋 5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数 loss
loss = tf.square(w + 1)
# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 生成会话，训练 40 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After %s steps: w is %f,   loss is %f.' % (i, w_val, loss_val))

'''C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_2/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 12:09:29.861381: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 12:09:30.631944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 12:09:30.637785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 12:09:31.206903: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 12:09:31.210280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 12:09:31.212412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 12:09:31.214693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
After 0 steps: w is 2.600000,   loss is 12.959999.
After 1 steps: w is 1.160000,   loss is 4.665599.
After 2 steps: w is 0.296000,   loss is 1.679616.
After 3 steps: w is -0.222400,   loss is 0.604662.
After 4 steps: w is -0.533440,   loss is 0.217678.
After 5 steps: w is -0.720064,   loss is 0.078364.
After 6 steps: w is -0.832038,   loss is 0.028211.
After 7 steps: w is -0.899223,   loss is 0.010156.
After 8 steps: w is -0.939534,   loss is 0.003656.
After 9 steps: w is -0.963720,   loss is 0.001316.
After 10 steps: w is -0.978232,   loss is 0.000474.
After 11 steps: w is -0.986939,   loss is 0.000171.
After 12 steps: w is -0.992164,   loss is 0.000061.
After 13 steps: w is -0.995298,   loss is 0.000022.
After 14 steps: w is -0.997179,   loss is 0.000008.
After 15 steps: w is -0.998307,   loss is 0.000003.
After 16 steps: w is -0.998984,   loss is 0.000001.
After 17 steps: w is -0.999391,   loss is 0.000000.
After 18 steps: w is -0.999634,   loss is 0.000000.
After 19 steps: w is -0.999781,   loss is 0.000000.
After 20 steps: w is -0.999868,   loss is 0.000000.
After 21 steps: w is -0.999921,   loss is 0.000000.
After 22 steps: w is -0.999953,   loss is 0.000000.
After 23 steps: w is -0.999972,   loss is 0.000000.
After 24 steps: w is -0.999983,   loss is 0.000000.
After 25 steps: w is -0.999990,   loss is 0.000000.
After 26 steps: w is -0.999994,   loss is 0.000000.
After 27 steps: w is -0.999996,   loss is 0.000000.
After 28 steps: w is -0.999998,   loss is 0.000000.
After 29 steps: w is -0.999999,   loss is 0.000000.
After 30 steps: w is -0.999999,   loss is 0.000000.
After 31 steps: w is -1.000000,   loss is 0.000000.
After 32 steps: w is -1.000000,   loss is 0.000000.
After 33 steps: w is -1.000000,   loss is 0.000000.
After 34 steps: w is -1.000000,   loss is 0.000000.
After 35 steps: w is -1.000000,   loss is 0.000000.
After 36 steps: w is -1.000000,   loss is 0.000000.
After 37 steps: w is -1.000000,   loss is 0.000000.
After 38 steps: w is -1.000000,   loss is 0.000000.
After 39 steps: w is -1.000000,   loss is 0.000000.

Process finished with exit code 0
'''
