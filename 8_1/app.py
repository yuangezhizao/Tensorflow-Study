#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/24 0024 13:31
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import utils
import vgg16
from Nclasses import labels


def main():
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.forward(images)
    while True:
        img_path = input('\nInput the path and image name: ')

        if img_path == '0':
            print('\nexiting-------------------------\n')
            exit(0)
        else:
            print('\nrunning-------------------------\n')
        img_ready = utils.load_image(img_path)

        fig = plt.figure(u'Top-5 预测结果')

        probability = sess.run(vgg.prob, feed_dict={images: img_ready})
        top5 = np.argsort(probability[0])[-1:-6:-1]
        print('top5:', top5)
        values = []
        bar_label = []
        for n, i in enumerate(top5):
            print('n:', n)
            print('i:', i)
            values.append(probability[0][i])
            bar_label.append(labels[i])
            print(i, ':', labels[i], '----', utils.percent(probability[0][i]))

        ax = fig.add_subplot(111)
        ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
        ax.set_ylabel(u'probabilityit')
        ax.set_title(u'Top-5')
        for a, b in zip(range(len(values)), values):
            ax.text(a, b + 0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
        plt.show()


if __name__ == '__main__':
    main()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/8_1/app.py
2019-03-24 17:37:13.017010: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-24 17:37:13.790978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-24 17:37:13.796633: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 17:37:14.354739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 17:37:14.358371: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 17:37:14.360542: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 17:37:14.363504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
build model started
time consuming: 4.305527

Input the path and image name: pic/0.jpg

running-------------------------

C:\Python37\lib\site-packages\skimage\transform\_warps.py:105: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
  warn("The default mode, 'constant', will be changed to 'reflect' in "
C:\Python37\lib\site-packages\skimage\transform\_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
2019-03-24 17:38:01.904854: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
2019-03-24 17:38:03.140793: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.05GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.175231: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 882.56MiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.190242: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.02GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.235144: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.05GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.292277: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.04GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.333041: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.07GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.391366: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.07GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.441797: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.14GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.451266: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.07GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-03-24 17:38:03.501327: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.07GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
top5: [19 13 10 20 16]
n: 0
i: 19
19 : chickadee ---- 100.00%
n: 1
i: 13
13 : junco
 snowbird ---- 0.00%
n: 2
i: 10
10 : brambling
 Fringilla montifringilla ---- 0.00%
n: 3
i: 20
20 : water ouzel
 dipper ---- 0.00%
n: 4
i: 16
16 : bulbul ---- 0.00%

Input the path and image name: 
Process finished with exit code -1
'''
