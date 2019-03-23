#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 10:56
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import tensorflow as tf

x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])

y = tf.matmul(x, w)

print(y)

with tf.Session() as sess:
    print(sess.run(y))
