#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 11:40
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 预测多或预测少的影响一样
# 0导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 1定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2定义损失函数及反向传播方法
# 定义损失函数为 MSE ,反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 3生成会话，训练 STEPS 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print('After %d training steps, w1 is: ' % (i))
            print(sess.run(w1), '\n')
    print('Final w1 is: \n', sess.run(w1))
# 在本代码 #2 中尝试其他反向传播方法，看对收敛速度的影响，把体会写到笔记中

# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_1/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 11:44:43.677767: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 11:44:44.520308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 11:44:44.526702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 11:44:45.117871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 11:44:45.121198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 11:44:45.123265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 11:44:45.125612: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 11:44:45.311080: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 0 training steps, w1 is: 
[[-0.80974597]
 [ 1.4852903 ]] 

After 500 training steps, w1 is: 
[[-0.46074435]
 [ 1.641878  ]] 

After 1000 training steps, w1 is: 
[[-0.21939856]
 [ 1.6984766 ]] 

After 1500 training steps, w1 is: 
[[-0.04415595]
 [ 1.7003176 ]] 

After 2000 training steps, w1 is: 
[[0.08942621]
 [1.673328  ]] 

After 2500 training steps, w1 is: 
[[0.19583553]
 [1.6322677 ]] 

After 3000 training steps, w1 is: 
[[0.28375748]
 [1.5854434 ]] 

After 3500 training steps, w1 is: 
[[0.35848638]
 [1.5374471 ]] 

After 4000 training steps, w1 is: 
[[0.4233252]
 [1.4907392]] 

After 4500 training steps, w1 is: 
[[0.48040032]
 [1.4465573 ]] 

After 5000 training steps, w1 is: 
[[0.5311361]
 [1.4054534]] 

After 5500 training steps, w1 is: 
[[0.57653254]
 [1.367594  ]] 

After 6000 training steps, w1 is: 
[[0.6173259]
 [1.3329402]] 

After 6500 training steps, w1 is: 
[[0.65408474]
 [1.3013425 ]] 

After 7000 training steps, w1 is: 
[[0.68726856]
 [1.2726018 ]] 

After 7500 training steps, w1 is: 
[[0.7172598]
 [1.2465004]] 

After 8000 training steps, w1 is: 
[[0.74438614]
 [1.2228196 ]] 

After 8500 training steps, w1 is: 
[[0.7689325]
 [1.2013482]] 

After 9000 training steps, w1 is: 
[[0.79115146]
 [1.1818888 ]] 

After 9500 training steps, w1 is: 
[[0.81126714]
 [1.1642567 ]] 

After 10000 training steps, w1 is: 
[[0.8294814]
 [1.1482829]] 

After 10500 training steps, w1 is: 
[[0.84597576]
 [1.1338127 ]] 

After 11000 training steps, w1 is: 
[[0.8609128]
 [1.1207061]] 

After 11500 training steps, w1 is: 
[[0.87444043]
 [1.1088346 ]] 

After 12000 training steps, w1 is: 
[[0.88669145]
 [1.0980824 ]] 

After 12500 training steps, w1 is: 
[[0.8977863]
 [1.0883439]] 

After 13000 training steps, w1 is: 
[[0.9078348]
 [1.0795243]] 

After 13500 training steps, w1 is: 
[[0.91693527]
 [1.0715363 ]] 

After 14000 training steps, w1 is: 
[[0.92517716]
 [1.0643018 ]] 

After 14500 training steps, w1 is: 
[[0.93264157]
 [1.0577497 ]] 

After 15000 training steps, w1 is: 
[[0.9394023]
 [1.0518153]] 

After 15500 training steps, w1 is: 
[[0.9455251]
 [1.0464406]] 

After 16000 training steps, w1 is: 
[[0.95107025]
 [1.0415728 ]] 

After 16500 training steps, w1 is: 
[[0.9560928]
 [1.037164 ]] 

After 17000 training steps, w1 is: 
[[0.96064115]
 [1.0331714 ]] 

After 17500 training steps, w1 is: 
[[0.96476096]
 [1.0295546 ]] 

After 18000 training steps, w1 is: 
[[0.9684917]
 [1.0262802]] 

After 18500 training steps, w1 is: 
[[0.9718707]
 [1.0233142]] 

After 19000 training steps, w1 is: 
[[0.974931 ]
 [1.0206276]] 

After 19500 training steps, w1 is: 
[[0.9777026]
 [1.0181949]] 

Final w1 is: 
 [[0.98019385]
 [1.0159807 ]]

Process finished with exit code 0
'''
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_1/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 11:46:42.167759: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 11:46:42.914104: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 11:46:42.919826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 11:46:43.442173: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 11:46:43.445534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 11:46:43.447625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 11:46:43.449870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 11:46:43.628338: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 0 training steps, w1 is: 
[[-0.80974597]
 [ 1.4852903 ]] 

After 500 training steps, w1 is: 
[[0.53368706]
 [1.4052336 ]] 

After 1000 training steps, w1 is: 
[[0.83336353]
 [1.1448474 ]] 

After 1500 training steps, w1 is: 
[[0.9420494]
 [1.0494746]] 

After 2000 training steps, w1 is: 
[[0.98163515]
 [1.0147355 ]] 

After 2500 training steps, w1 is: 
[[0.9960538]
 [1.0020828]] 

After 3000 training steps, w1 is: 
[[1.0013052]
 [0.997474 ]] 

After 3500 training steps, w1 is: 
[[1.0032176]
 [0.9957958]] 

After 4000 training steps, w1 is: 
[[1.0039146]
 [0.9951844]] 

After 4500 training steps, w1 is: 
[[1.0041676 ]
 [0.99496204]] 

After 5000 training steps, w1 is: 
[[1.0042601]
 [0.9948809]] 

After 5500 training steps, w1 is: 
[[1.004293  ]
 [0.99485284]] 

After 6000 training steps, w1 is: 
[[1.0043079 ]
 [0.99483955]] 

After 6500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 7000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 7500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 8000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 8500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 9000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 9500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 10000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 10500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 11000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 11500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 12000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 12500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 13000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 13500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 14000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 14500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 15000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 15500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 16000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 16500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 17000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 17500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 18000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 18500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 19000 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

After 19500 training steps, w1 is: 
[[1.0043106 ]
 [0.99483734]] 

Final w1 is: 
 [[1.0043069]
 [0.9948299]]

Process finished with exit code 0
'''
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)
'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_1/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 11:47:40.248294: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 11:47:40.988146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 11:47:40.994290: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 11:47:41.508857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 11:47:41.512196: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 11:47:41.514406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 11:47:41.516702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 11:47:41.708972: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 0 training steps, w1 is: 
[[-0.81031823]
 [ 1.4855988 ]] 

After 500 training steps, w1 is: 
[[-0.43049008]
 [ 1.789822  ]] 

After 1000 training steps, w1 is: 
[[-0.13943434]
 [ 1.7868288 ]] 

After 1500 training steps, w1 is: 
[[0.10796265]
 [1.6586488 ]] 

After 2000 training steps, w1 is: 
[[0.3273534]
 [1.5091106]] 

After 2500 training steps, w1 is: 
[[0.5191771]
 [1.3682904]] 

After 3000 training steps, w1 is: 
[[0.68027073]
 [1.2463676 ]] 

After 3500 training steps, w1 is: 
[[0.80737233]
 [1.1485661 ]] 

After 4000 training steps, w1 is: 
[[0.8988763]
 [1.0774578]] 

After 4500 training steps, w1 is: 
[[0.9566475]
 [1.0322955]] 

After 5000 training steps, w1 is: 
[[0.98706496]
 [1.0084281 ]] 

After 5500 training steps, w1 is: 
[[0.9996404 ]
 [0.99853456]] 

After 6000 training steps, w1 is: 
[[1.0034434 ]
 [0.99553627]] 

After 6500 training steps, w1 is: 
[[1.0042175 ]
 [0.99492735]] 

After 7000 training steps, w1 is: 
[[1.0043159]
 [0.9948547]] 

After 7500 training steps, w1 is: 
[[1.0043254 ]
 [0.99485356]] 

After 8000 training steps, w1 is: 
[[1.0043299]
 [0.9948566]] 

After 8500 training steps, w1 is: 
[[1.0043344 ]
 [0.99485934]] 

After 9000 training steps, w1 is: 
[[1.0043393]
 [0.9948613]] 

After 9500 training steps, w1 is: 
[[1.0043442]
 [0.994863 ]] 

After 10000 training steps, w1 is: 
[[1.0043497]
 [0.9948635]] 

After 10500 training steps, w1 is: 
[[1.0043544]
 [0.9948638]] 

After 11000 training steps, w1 is: 
[[1.0043578 ]
 [0.99486405]] 

After 11500 training steps, w1 is: 
[[1.0043604]
 [0.9948643]] 

After 12000 training steps, w1 is: 
[[1.0043623]
 [0.9948643]] 

After 12500 training steps, w1 is: 
[[1.004364  ]
 [0.99486387]] 

After 13000 training steps, w1 is: 
[[1.004365 ]
 [0.9948636]] 

After 13500 training steps, w1 is: 
[[1.0043657]
 [0.9948636]] 

After 14000 training steps, w1 is: 
[[1.0043657]
 [0.994864 ]] 

After 14500 training steps, w1 is: 
[[1.0043665 ]
 [0.99486333]] 

After 15000 training steps, w1 is: 
[[1.0043665 ]
 [0.99486357]] 

After 15500 training steps, w1 is: 
[[1.004366  ]
 [0.99486387]] 

After 16000 training steps, w1 is: 
[[1.0043665]
 [0.9948637]] 

After 16500 training steps, w1 is: 
[[1.0043665]
 [0.9948637]] 

After 17000 training steps, w1 is: 
[[1.0043665]
 [0.9948637]] 

After 17500 training steps, w1 is: 
[[1.0043662]
 [0.9948638]] 

After 18000 training steps, w1 is: 
[[1.0043662 ]
 [0.99486387]] 

After 18500 training steps, w1 is: 
[[1.0043662 ]
 [0.99486387]] 

After 19000 training steps, w1 is: 
[[1.0043662 ]
 [0.99486387]] 

After 19500 training steps, w1 is: 
[[1.0043662 ]
 [0.99486387]] 

Final w1 is: 
 [[1.0043191]
 [0.9948099]]

Process finished with exit code 0
'''
