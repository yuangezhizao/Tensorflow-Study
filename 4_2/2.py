#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 12:14
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 设损失函数 loss=(w+1)^2, 令 w 初值是常数 10。反向传播就是求最优 w，即求最小 loss 对应的 w 值
# 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛度
import tensorflow as tf

LEARNING_RATE_BASE = 0.1  # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 喂入多少轮 BATCH_SIZE 后，更新一次学习率，一般设为：总样本数 / BATCH_SIZE

# 运行了几轮 BATCH_SIZE 的计数器，初值给 0, 设为不被训练
global_step = tf.Variable(0, trainable=False)
# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                           staircase=True)
# 定义待优化参数，初值给 10
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数 loss
loss = tf.square(w + 1)
# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 生成会话，训练 40 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f' % (
            i, global_step_val, w_val, learning_rate_val, loss_val))

'''C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_2/2.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 12:16:03.942796: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 12:16:04.681918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 12:16:04.687760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 12:16:05.218313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 12:16:05.221764: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 12:16:05.223719: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 12:16:05.226097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
After 0 steps: global_step is 1.000000, w is 3.800000, learning rate is 0.099000, loss is 23.040001
After 1 steps: global_step is 2.000000, w is 2.849600, learning rate is 0.098010, loss is 14.819419
After 2 steps: global_step is 3.000000, w is 2.095001, learning rate is 0.097030, loss is 9.579033
After 3 steps: global_step is 4.000000, w is 1.494386, learning rate is 0.096060, loss is 6.221960
After 4 steps: global_step is 5.000000, w is 1.015166, learning rate is 0.095099, loss is 4.060895
After 5 steps: global_step is 6.000000, w is 0.631886, learning rate is 0.094148, loss is 2.663051
After 6 steps: global_step is 7.000000, w is 0.324608, learning rate is 0.093207, loss is 1.754587
After 7 steps: global_step is 8.000000, w is 0.077684, learning rate is 0.092274, loss is 1.161402
After 8 steps: global_step is 9.000000, w is -0.121202, learning rate is 0.091352, loss is 0.772287
After 9 steps: global_step is 10.000000, w is -0.281761, learning rate is 0.090438, loss is 0.515867
After 10 steps: global_step is 11.000000, w is -0.411674, learning rate is 0.089534, loss is 0.346128
After 11 steps: global_step is 12.000000, w is -0.517024, learning rate is 0.088638, loss is 0.233266
After 12 steps: global_step is 13.000000, w is -0.602644, learning rate is 0.087752, loss is 0.157891
After 13 steps: global_step is 14.000000, w is -0.672382, learning rate is 0.086875, loss is 0.107334
After 14 steps: global_step is 15.000000, w is -0.729305, learning rate is 0.086006, loss is 0.073276
After 15 steps: global_step is 16.000000, w is -0.775868, learning rate is 0.085146, loss is 0.050235
After 16 steps: global_step is 17.000000, w is -0.814036, learning rate is 0.084294, loss is 0.034583
After 17 steps: global_step is 18.000000, w is -0.845387, learning rate is 0.083451, loss is 0.023905
After 18 steps: global_step is 19.000000, w is -0.871193, learning rate is 0.082617, loss is 0.016591
After 19 steps: global_step is 20.000000, w is -0.892476, learning rate is 0.081791, loss is 0.011561
After 20 steps: global_step is 21.000000, w is -0.910065, learning rate is 0.080973, loss is 0.008088
After 21 steps: global_step is 22.000000, w is -0.924629, learning rate is 0.080163, loss is 0.005681
After 22 steps: global_step is 23.000000, w is -0.936713, learning rate is 0.079361, loss is 0.004005
After 23 steps: global_step is 24.000000, w is -0.946758, learning rate is 0.078568, loss is 0.002835
After 24 steps: global_step is 25.000000, w is -0.955125, learning rate is 0.077782, loss is 0.002014
After 25 steps: global_step is 26.000000, w is -0.962106, learning rate is 0.077004, loss is 0.001436
After 26 steps: global_step is 27.000000, w is -0.967942, learning rate is 0.076234, loss is 0.001028
After 27 steps: global_step is 28.000000, w is -0.972830, learning rate is 0.075472, loss is 0.000738
After 28 steps: global_step is 29.000000, w is -0.976931, learning rate is 0.074717, loss is 0.000532
After 29 steps: global_step is 30.000000, w is -0.980378, learning rate is 0.073970, loss is 0.000385
After 30 steps: global_step is 31.000000, w is -0.983281, learning rate is 0.073230, loss is 0.000280
After 31 steps: global_step is 32.000000, w is -0.985730, learning rate is 0.072498, loss is 0.000204
After 32 steps: global_step is 33.000000, w is -0.987799, learning rate is 0.071773, loss is 0.000149
After 33 steps: global_step is 34.000000, w is -0.989550, learning rate is 0.071055, loss is 0.000109
After 34 steps: global_step is 35.000000, w is -0.991035, learning rate is 0.070345, loss is 0.000080
After 35 steps: global_step is 36.000000, w is -0.992297, learning rate is 0.069641, loss is 0.000059
After 36 steps: global_step is 37.000000, w is -0.993369, learning rate is 0.068945, loss is 0.000044
After 37 steps: global_step is 38.000000, w is -0.994284, learning rate is 0.068255, loss is 0.000033
After 38 steps: global_step is 39.000000, w is -0.995064, learning rate is 0.067573, loss is 0.000024
After 39 steps: global_step is 40.000000, w is -0.995731, learning rate is 0.066897, loss is 0.000018

Process finished with exit code 0
'''
