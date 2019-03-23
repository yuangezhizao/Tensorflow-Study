#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 12:20
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import tensorflow as tf

# 1. 定义变量及滑动平均类
# 定义一个 32 位浮点变量，初始值为 0.0 这个代码就是不断更新 w1 参数，优化 w1 参数，滑动平均做了个 w1 的影子
w1 = tf.Variable(0, dtype=tf.float32)
# 定义 num_updates（NN 的迭代轮数）,初始值为 0，不可被优化（训练），这个参数不训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，给衰减率为 0.99，当前轮数 global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# ema.apply 后的括号里是更新列表，每次运行 sess.run（ema_op） 时，对更新列表中的元素求滑动平均值
# 在实际应用中会使用 tf.trainable_variables() 自动将所有待训练的参数汇总为列表
# ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

# 2. 查看不同迭代中变量取值的变化
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 用 ema.average(w1) 获取 w1 滑动平均值（要运行多个节点，作为列表中的元素列出，写在 sess.run 中）
    # 打印出当前参数 w1 和 w1 滑动平均值
    print('current global_step:', sess.run(global_step))
    print('current w1', sess.run([w1, ema.average(w1)]))

    # 参数 w1 的值赋为 1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1', sess.run([w1, ema.average(w1)]))

    # 更新 global_step 和 w1 的值，模拟出轮数为 100 时，参数 w1 变为 10，以下代码 global_step 保持为 100，每次执行滑动平均操作，影子值会更新
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    # 每次 sess.run 会更新一次 w1 的滑动平均值
    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print('current global_step:', sess.run(global_step))
    print('current w1:', sess.run([w1, ema.average(w1)]))

# 更改 MOVING_AVERAGE_DECAY 为 0.1，看影子追随速度

'''C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_3/1.py
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-23 12:28:56.843492: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 12:28:57.619403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 12:28:57.625501: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 12:28:58.201418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 12:28:58.204804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 12:28:58.207220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 12:28:58.209429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
current global_step: 0
current w1 [0.0, 0.0]
current global_step: 0
current w1 [1.0, 0.9]
current global_step: 100
current w1: [10.0, 1.6445453]
current global_step: 100
current w1: [10.0, 2.3281732]
current global_step: 100
current w1: [10.0, 2.955868]
current global_step: 100
current w1: [10.0, 3.532206]
current global_step: 100
current w1: [10.0, 4.061389]
current global_step: 100
current w1: [10.0, 4.547275]
current global_step: 100
current w1: [10.0, 4.9934072]

Process finished with exit code 0
'''
