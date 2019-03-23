#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 11:05
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 两层简单神经网络（全连接）
import tensorflow as tf

# 定义输入和参数
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/3_2/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 11:14:42.239277: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 11:14:42.988404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 11:14:42.994129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 11:14:43.505239: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 11:14:43.508674: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 11:14:43.510765: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 11:14:43.513035: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 11:14:43.681746: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
[[3.0904665]]

Process finished with exit code 0
'''
