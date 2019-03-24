#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/24 0024 10:09
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import mnist_backward
import mnist_forward
import numpy as np
import tensorflow as tf
from PIL import Image


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print('No checkpoint file found')
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application():
    testNum = input('input the number of test pictures:')
    for i in range(int(testNum)):
        testPic = input('the path of test picture:')
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print('The prediction number is:', preValue)


def main():
    application()


if __name__ == '__main__':
    main()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/6_1/mnist_app.py
input the number of test pictures:10
the path of test picture:pic/0.png
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-03-24 10:25:23.892048: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-24 10:25:24.633338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-24 10:25:24.639221: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:25:25.148356: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:25:25.151620: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:25:25.153612: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:25:25.155818: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.
2019-03-24 10:25:25.334677: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
The prediction number is: [0]
the path of test picture:pic/1.png
2019-03-24 10:25:38.817301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:25:38.819916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:25:38.823273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:25:38.825291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:25:38.827606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [1]
the path of test picture:pic/2.png
2019-03-24 10:25:52.034879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:25:52.037645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:25:52.041006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:25:52.043054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:25:52.045184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [2]
the path of test picture:pic/3.png
2019-03-24 10:26:03.361310: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:26:03.363815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:26:03.367191: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:26:03.369262: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:26:03.371388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [3]
the path of test picture:pic/4.png
2019-03-24 10:33:31.130138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:31.312411: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:31.315695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:31.317742: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:31.320237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [4]
the path of test picture:pic/5.png
2019-03-24 10:33:35.082587: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:35.085230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:35.088992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:35.091060: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:35.093224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [5]
the path of test picture:pic/6.png
2019-03-24 10:33:38.084916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:38.087521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:38.090874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:38.092917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:38.095046: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [6]
the path of test picture:pic/7.png
2019-03-24 10:33:40.972819: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:40.975472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:40.978766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:40.980785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:40.983065: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [7]
the path of test picture:pic/8.png
2019-03-24 10:33:43.733133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:43.735663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:43.739028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:43.741032: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:43.743135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [8]
the path of test picture:pic/9.png
2019-03-24 10:33:47.656383: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 10:33:47.659116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 10:33:47.662428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 10:33:47.664458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 10:33:47.666573: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
The prediction number is: [9]

Process finished with exit code 0
'''
