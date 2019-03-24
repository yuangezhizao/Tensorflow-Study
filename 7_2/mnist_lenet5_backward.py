#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/24 0024 12:41
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import os

import mnist_lenet5_forward
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    y = mnist_lenet5_forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/7_2/mnist_lenet5_backward.py
WARNING:tensorflow:From D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/7_2/mnist_lenet5_backward.py:77: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please use urllib or similar directly.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Please use tf.data to implement this functionality.
Extracting ./data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting ./data/train-labels-idx1-ubyte.gz
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting ./data/t10k-images-idx3-ubyte.gz
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
Extracting ./data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From D:\yuangezhizao\Documents\PycharmProjects\Tensorflow-Study\7_2\mnist_lenet5_forward.py:60: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
2019-03-24 13:38:35.050545: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-24 13:38:35.836862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-24 13:38:35.842845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-24 13:38:36.448638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-24 13:38:36.452359: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-24 13:38:36.454497: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-24 13:38:36.456774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1372 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-24 13:38:36.962878: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 1 training step(s), loss on training batch is 6.26063.
After 101 training step(s), loss on training batch is 1.92559.
After 201 training step(s), loss on training batch is 1.5969.
After 301 training step(s), loss on training batch is 1.47031.
After 401 training step(s), loss on training batch is 1.31606.
After 501 training step(s), loss on training batch is 1.13109.
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\training\saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.
After 601 training step(s), loss on training batch is 1.04185.
After 701 training step(s), loss on training batch is 1.04572.
After 801 training step(s), loss on training batch is 0.999314.
After 901 training step(s), loss on training batch is 1.20164.
After 1001 training step(s), loss on training batch is 0.931952.
After 1101 training step(s), loss on training batch is 0.927714.
After 1201 training step(s), loss on training batch is 1.06095.
After 1301 training step(s), loss on training batch is 0.763744.
After 1401 training step(s), loss on training batch is 1.03923.
After 1501 training step(s), loss on training batch is 0.880137.
After 1601 training step(s), loss on training batch is 0.898442.
After 1701 training step(s), loss on training batch is 0.903217.
After 1801 training step(s), loss on training batch is 0.901716.
After 1901 training step(s), loss on training batch is 0.85283.
After 2001 training step(s), loss on training batch is 0.796007.
After 2101 training step(s), loss on training batch is 0.791778.
After 2201 training step(s), loss on training batch is 0.862247.
After 2301 training step(s), loss on training batch is 0.759797.
After 2401 training step(s), loss on training batch is 0.739344.
After 2501 training step(s), loss on training batch is 0.931866.
After 2601 training step(s), loss on training batch is 0.822988.
After 2701 training step(s), loss on training batch is 0.804365.
After 2801 training step(s), loss on training batch is 0.764375.
After 2901 training step(s), loss on training batch is 0.771429.
After 3001 training step(s), loss on training batch is 0.966594.
After 3101 training step(s), loss on training batch is 0.852383.
After 3201 training step(s), loss on training batch is 0.766062.
After 3301 training step(s), loss on training batch is 0.820025.
After 3401 training step(s), loss on training batch is 0.918287.
After 3501 training step(s), loss on training batch is 0.727936.
After 3601 training step(s), loss on training batch is 0.769309.
After 3701 training step(s), loss on training batch is 0.76396.
After 3801 training step(s), loss on training batch is 0.763146.
After 3901 training step(s), loss on training batch is 0.772036.
After 4001 training step(s), loss on training batch is 0.821111.
After 4101 training step(s), loss on training batch is 0.767625.
After 4201 training step(s), loss on training batch is 0.948026.
After 4301 training step(s), loss on training batch is 0.873042.
After 4401 training step(s), loss on training batch is 0.760312.
After 4501 training step(s), loss on training batch is 0.872279.
After 4601 training step(s), loss on training batch is 0.720263.
After 4701 training step(s), loss on training batch is 0.817477.
After 4801 training step(s), loss on training batch is 0.830749.
After 4901 training step(s), loss on training batch is 0.880379.
After 5001 training step(s), loss on training batch is 0.810555.
After 5101 training step(s), loss on training batch is 0.754946.
After 5201 training step(s), loss on training batch is 0.763367.
After 5301 training step(s), loss on training batch is 0.86674.
After 5401 training step(s), loss on training batch is 0.836874.
After 5501 training step(s), loss on training batch is 0.757257.
After 5601 training step(s), loss on training batch is 0.755645.
After 5701 training step(s), loss on training batch is 0.793618.
After 5801 training step(s), loss on training batch is 0.815984.
After 5901 training step(s), loss on training batch is 0.76014.
After 6001 training step(s), loss on training batch is 0.760627.
After 6101 training step(s), loss on training batch is 0.743822.
After 6201 training step(s), loss on training batch is 0.840639.
After 6301 training step(s), loss on training batch is 0.742182.
After 6401 training step(s), loss on training batch is 0.757858.
After 6501 training step(s), loss on training batch is 0.730707.
After 6601 training step(s), loss on training batch is 0.868041.
After 6701 training step(s), loss on training batch is 0.713239.
After 6801 training step(s), loss on training batch is 0.764353.
After 6901 training step(s), loss on training batch is 0.766517.
After 7001 training step(s), loss on training batch is 0.736728.
After 7101 training step(s), loss on training batch is 0.754941.
After 7201 training step(s), loss on training batch is 0.722469.
After 7301 training step(s), loss on training batch is 0.752093.
After 7401 training step(s), loss on training batch is 0.756027.
After 7501 training step(s), loss on training batch is 0.863544.
After 7601 training step(s), loss on training batch is 0.758512.
After 7701 training step(s), loss on training batch is 0.69823.
After 7801 training step(s), loss on training batch is 0.704393.
After 7901 training step(s), loss on training batch is 0.686168.
After 8001 training step(s), loss on training batch is 0.695188.
After 8101 training step(s), loss on training batch is 0.800707.
After 8201 training step(s), loss on training batch is 0.697372.
After 8301 training step(s), loss on training batch is 0.688759.
After 8401 training step(s), loss on training batch is 0.733643.
After 8501 training step(s), loss on training batch is 0.814063.
After 8601 training step(s), loss on training batch is 0.754083.
After 8701 training step(s), loss on training batch is 0.699687.
After 8801 training step(s), loss on training batch is 0.785507.
After 8901 training step(s), loss on training batch is 0.746603.
After 9001 training step(s), loss on training batch is 0.731439.
After 9101 training step(s), loss on training batch is 0.801787.
After 9201 training step(s), loss on training batch is 0.735606.
After 9301 training step(s), loss on training batch is 0.804531.
After 9401 training step(s), loss on training batch is 0.685068.
After 9501 training step(s), loss on training batch is 0.904594.
After 9601 training step(s), loss on training batch is 0.726698.
After 9701 training step(s), loss on training batch is 0.672878.
After 9801 training step(s), loss on training batch is 0.746725.
After 9901 training step(s), loss on training batch is 0.731782.
After 10001 training step(s), loss on training batch is 0.818848.
After 10101 training step(s), loss on training batch is 0.715777.
After 10201 training step(s), loss on training batch is 0.76293.
After 10301 training step(s), loss on training batch is 0.722475.
After 10401 training step(s), loss on training batch is 0.665627.
After 10501 training step(s), loss on training batch is 0.718027.
After 10601 training step(s), loss on training batch is 0.69324.
After 10701 training step(s), loss on training batch is 0.693716.
After 10801 training step(s), loss on training batch is 0.713539.
After 10901 training step(s), loss on training batch is 0.727114.
After 11001 training step(s), loss on training batch is 0.702183.
After 11101 training step(s), loss on training batch is 0.682742.
After 11201 training step(s), loss on training batch is 0.671438.
After 11301 training step(s), loss on training batch is 0.736906.
After 11401 training step(s), loss on training batch is 0.745051.
After 11501 training step(s), loss on training batch is 0.722482.
After 11601 training step(s), loss on training batch is 0.746543.
After 11701 training step(s), loss on training batch is 0.68168.
After 11801 training step(s), loss on training batch is 0.677532.
After 11901 training step(s), loss on training batch is 0.769755.
After 12001 training step(s), loss on training batch is 0.699192.
After 12101 training step(s), loss on training batch is 0.660505.
After 12201 training step(s), loss on training batch is 0.685654.
After 12301 training step(s), loss on training batch is 0.759417.
After 12401 training step(s), loss on training batch is 0.707114.
After 12501 training step(s), loss on training batch is 0.815164.
After 12601 training step(s), loss on training batch is 0.78678.
After 12701 training step(s), loss on training batch is 0.694717.
After 12801 training step(s), loss on training batch is 0.681192.
After 12901 training step(s), loss on training batch is 0.677004.
After 13001 training step(s), loss on training batch is 0.690643.
After 13101 training step(s), loss on training batch is 0.663546.
After 13201 training step(s), loss on training batch is 0.733891.
After 13301 training step(s), loss on training batch is 0.75797.
After 13401 training step(s), loss on training batch is 0.678075.
After 13501 training step(s), loss on training batch is 0.788116.
After 13601 training step(s), loss on training batch is 0.752975.
After 13701 training step(s), loss on training batch is 0.684452.
After 13801 training step(s), loss on training batch is 0.762565.
After 13901 training step(s), loss on training batch is 0.653371.
After 14001 training step(s), loss on training batch is 0.680332.
After 14101 training step(s), loss on training batch is 0.729231.
After 14201 training step(s), loss on training batch is 0.698859.
After 14301 training step(s), loss on training batch is 0.69098.
After 14401 training step(s), loss on training batch is 0.672296.
After 14501 training step(s), loss on training batch is 0.667837.
After 14601 training step(s), loss on training batch is 0.678778.
After 14701 training step(s), loss on training batch is 0.706525.
After 14801 training step(s), loss on training batch is 0.760393.
After 14901 training step(s), loss on training batch is 0.673431.
After 15001 training step(s), loss on training batch is 0.725244.
After 15101 training step(s), loss on training batch is 0.797518.
After 15201 training step(s), loss on training batch is 0.716795.
After 15301 training step(s), loss on training batch is 0.678848.
After 15401 training step(s), loss on training batch is 0.66115.
After 15501 training step(s), loss on training batch is 0.654078.
After 15601 training step(s), loss on training batch is 0.674892.
After 15701 training step(s), loss on training batch is 0.708834.
After 15801 training step(s), loss on training batch is 0.69218.
After 15901 training step(s), loss on training batch is 0.652918.
After 16001 training step(s), loss on training batch is 0.778921.
After 16101 training step(s), loss on training batch is 0.718423.
After 16201 training step(s), loss on training batch is 0.763053.
After 16301 training step(s), loss on training batch is 0.713086.
After 16401 training step(s), loss on training batch is 0.772468.
After 16501 training step(s), loss on training batch is 0.722951.
After 16601 training step(s), loss on training batch is 0.735263.
After 16701 training step(s), loss on training batch is 0.69567.
After 16801 training step(s), loss on training batch is 0.675046.
After 16901 training step(s), loss on training batch is 0.662051.
After 17001 training step(s), loss on training batch is 0.693966.
After 17101 training step(s), loss on training batch is 0.721596.
After 17201 training step(s), loss on training batch is 0.677455.
After 17301 training step(s), loss on training batch is 0.749906.
After 17401 training step(s), loss on training batch is 0.681242.
After 17501 training step(s), loss on training batch is 0.710037.
After 17601 training step(s), loss on training batch is 0.67259.
After 17701 training step(s), loss on training batch is 0.758855.
After 17801 training step(s), loss on training batch is 0.68984.
After 17901 training step(s), loss on training batch is 0.668615.
After 18001 training step(s), loss on training batch is 0.716558.
After 18101 training step(s), loss on training batch is 0.699102.
After 18201 training step(s), loss on training batch is 0.691428.
After 18301 training step(s), loss on training batch is 0.669582.
After 18401 training step(s), loss on training batch is 0.745403.
After 18501 training step(s), loss on training batch is 0.686434.
After 18601 training step(s), loss on training batch is 0.750614.
After 18701 training step(s), loss on training batch is 0.689958.
After 18801 training step(s), loss on training batch is 0.740874.
After 18901 training step(s), loss on training batch is 0.762757.
After 19001 training step(s), loss on training batch is 0.695113.
After 19101 training step(s), loss on training batch is 0.704648.
After 19201 training step(s), loss on training batch is 0.67735.
After 19301 training step(s), loss on training batch is 0.709337.
After 19401 training step(s), loss on training batch is 0.721933.
After 19501 training step(s), loss on training batch is 0.661036.
After 19601 training step(s), loss on training batch is 0.698044.
After 19701 training step(s), loss on training batch is 0.666259.
After 19801 training step(s), loss on training batch is 0.680187.
After 19901 training step(s), loss on training batch is 0.649084.
After 20001 training step(s), loss on training batch is 0.652215.
After 20101 training step(s), loss on training batch is 0.697839.
After 20201 training step(s), loss on training batch is 0.652687.
After 20301 training step(s), loss on training batch is 0.74217.
After 20401 training step(s), loss on training batch is 0.744419.
After 20501 training step(s), loss on training batch is 0.728433.
After 20601 training step(s), loss on training batch is 0.744894.
After 20701 training step(s), loss on training batch is 0.70947.
After 20801 training step(s), loss on training batch is 0.658006.
After 20901 training step(s), loss on training batch is 0.689027.
After 21001 training step(s), loss on training batch is 0.684234.
After 21101 training step(s), loss on training batch is 0.70936.
After 21201 training step(s), loss on training batch is 0.768082.
After 21301 training step(s), loss on training batch is 0.744963.
After 21401 training step(s), loss on training batch is 0.728966.
After 21501 training step(s), loss on training batch is 0.70764.
After 21601 training step(s), loss on training batch is 0.742818.
After 21701 training step(s), loss on training batch is 0.676436.
After 21801 training step(s), loss on training batch is 0.661339.
After 21901 training step(s), loss on training batch is 0.725796.
After 22001 training step(s), loss on training batch is 0.690815.
After 22101 training step(s), loss on training batch is 0.708954.
After 22201 training step(s), loss on training batch is 0.679728.
After 22301 training step(s), loss on training batch is 0.708833.
After 22401 training step(s), loss on training batch is 0.740076.
After 22501 training step(s), loss on training batch is 0.651307.
After 22601 training step(s), loss on training batch is 0.682813.
After 22701 training step(s), loss on training batch is 0.739885.
After 22801 training step(s), loss on training batch is 0.656614.
After 22901 training step(s), loss on training batch is 0.683406.
After 23001 training step(s), loss on training batch is 0.67311.
After 23101 training step(s), loss on training batch is 0.668158.
After 23201 training step(s), loss on training batch is 0.68001.
After 23301 training step(s), loss on training batch is 0.733193.
After 23401 training step(s), loss on training batch is 0.714991.
After 23501 training step(s), loss on training batch is 0.698613.
After 23601 training step(s), loss on training batch is 0.667261.
After 23701 training step(s), loss on training batch is 0.689944.
After 23801 training step(s), loss on training batch is 0.642274.
After 23901 training step(s), loss on training batch is 0.690601.
After 24001 training step(s), loss on training batch is 0.756464.
After 24101 training step(s), loss on training batch is 0.682733.
After 24201 training step(s), loss on training batch is 0.666618.
After 24301 training step(s), loss on training batch is 0.654211.
After 24401 training step(s), loss on training batch is 0.686472.
After 24501 training step(s), loss on training batch is 0.700508.
After 24601 training step(s), loss on training batch is 0.673081.
After 24701 training step(s), loss on training batch is 0.719474.
After 24801 training step(s), loss on training batch is 0.661487.
After 24901 training step(s), loss on training batch is 0.689011.
After 25001 training step(s), loss on training batch is 0.699471.
After 25101 training step(s), loss on training batch is 0.642751.
After 25201 training step(s), loss on training batch is 0.659476.
After 25301 training step(s), loss on training batch is 0.684647.
After 25401 training step(s), loss on training batch is 0.734408.
After 25501 training step(s), loss on training batch is 0.674996.
After 25601 training step(s), loss on training batch is 0.685281.
After 25701 training step(s), loss on training batch is 0.641909.
After 25801 training step(s), loss on training batch is 0.690967.
After 25901 training step(s), loss on training batch is 0.728869.
After 26001 training step(s), loss on training batch is 0.673707.
After 26101 training step(s), loss on training batch is 0.670012.
After 26201 training step(s), loss on training batch is 0.762629.
After 26301 training step(s), loss on training batch is 0.686216.
After 26401 training step(s), loss on training batch is 0.707061.
After 26501 training step(s), loss on training batch is 0.690193.
After 26601 training step(s), loss on training batch is 0.650643.
After 26701 training step(s), loss on training batch is 0.660919.
After 26801 training step(s), loss on training batch is 0.676286.
After 26901 training step(s), loss on training batch is 0.651096.
After 27001 training step(s), loss on training batch is 0.672308.
After 27101 training step(s), loss on training batch is 0.725018.
After 27201 training step(s), loss on training batch is 0.64885.
After 27301 training step(s), loss on training batch is 0.703086.
After 27401 training step(s), loss on training batch is 0.66959.
After 27501 training step(s), loss on training batch is 0.662104.
After 27601 training step(s), loss on training batch is 0.665781.
After 27701 training step(s), loss on training batch is 0.650642.
After 27801 training step(s), loss on training batch is 0.662383.
After 27901 training step(s), loss on training batch is 0.655451.
After 28001 training step(s), loss on training batch is 0.712072.
After 28101 training step(s), loss on training batch is 0.681644.
After 28201 training step(s), loss on training batch is 0.681517.
After 28301 training step(s), loss on training batch is 0.692985.
After 28401 training step(s), loss on training batch is 0.649383.
After 28501 training step(s), loss on training batch is 0.814577.
After 28601 training step(s), loss on training batch is 0.745507.
After 28701 training step(s), loss on training batch is 0.751599.
After 28801 training step(s), loss on training batch is 0.663456.
After 28901 training step(s), loss on training batch is 0.650208.
After 29001 training step(s), loss on training batch is 0.683607.
After 29101 training step(s), loss on training batch is 0.663391.
After 29201 training step(s), loss on training batch is 0.775576.
After 29301 training step(s), loss on training batch is 0.683234.
After 29401 training step(s), loss on training batch is 0.66356.
After 29501 training step(s), loss on training batch is 0.679506.
After 29601 training step(s), loss on training batch is 0.748981.
After 29701 training step(s), loss on training batch is 0.784089.
After 29801 training step(s), loss on training batch is 0.659348.
After 29901 training step(s), loss on training batch is 0.685274.
After 30001 training step(s), loss on training batch is 0.69903.
After 30101 training step(s), loss on training batch is 0.658362.
After 30201 training step(s), loss on training batch is 0.652756.
After 30301 training step(s), loss on training batch is 0.649341.
After 30401 training step(s), loss on training batch is 0.671464.
After 30501 training step(s), loss on training batch is 0.686293.
After 30601 training step(s), loss on training batch is 0.644304.
After 30701 training step(s), loss on training batch is 0.643991.
After 30801 training step(s), loss on training batch is 0.696113.
After 30901 training step(s), loss on training batch is 0.716302.
After 31001 training step(s), loss on training batch is 0.649078.
After 31101 training step(s), loss on training batch is 0.712249.
After 31201 training step(s), loss on training batch is 0.679263.
After 31301 training step(s), loss on training batch is 0.708385.
After 31401 training step(s), loss on training batch is 0.70421.
After 31501 training step(s), loss on training batch is 0.725349.
After 31601 training step(s), loss on training batch is 0.662414.
After 31701 training step(s), loss on training batch is 0.65892.
After 31801 training step(s), loss on training batch is 0.651511.
After 31901 training step(s), loss on training batch is 0.744218.
After 32001 training step(s), loss on training batch is 0.658504.
After 32101 training step(s), loss on training batch is 0.653794.
After 32201 training step(s), loss on training batch is 0.640236.
After 32301 training step(s), loss on training batch is 0.651455.
After 32401 training step(s), loss on training batch is 0.636254.
After 32501 training step(s), loss on training batch is 0.660241.
After 32601 training step(s), loss on training batch is 0.832194.
After 32701 training step(s), loss on training batch is 0.703707.
After 32801 training step(s), loss on training batch is 0.711997.
After 32901 training step(s), loss on training batch is 0.685504.
After 33001 training step(s), loss on training batch is 0.639031.
After 33101 training step(s), loss on training batch is 0.76072.
After 33201 training step(s), loss on training batch is 0.681807.
After 33301 training step(s), loss on training batch is 0.642017.
After 33401 training step(s), loss on training batch is 0.764957.
After 33501 training step(s), loss on training batch is 0.650992.
After 33601 training step(s), loss on training batch is 0.71216.
After 33701 training step(s), loss on training batch is 0.748103.
After 33801 training step(s), loss on training batch is 0.660933.
After 33901 training step(s), loss on training batch is 0.702675.
After 34001 training step(s), loss on training batch is 0.662165.
After 34101 training step(s), loss on training batch is 0.653599.
After 34201 training step(s), loss on training batch is 0.74392.
After 34301 training step(s), loss on training batch is 0.691442.
After 34401 training step(s), loss on training batch is 0.721786.
After 34501 training step(s), loss on training batch is 0.645221.
After 34601 training step(s), loss on training batch is 0.662918.
After 34701 training step(s), loss on training batch is 0.651748.
After 34801 training step(s), loss on training batch is 0.692231.
After 34901 training step(s), loss on training batch is 0.661331.
After 35001 training step(s), loss on training batch is 0.660335.
After 35101 training step(s), loss on training batch is 0.665055.
After 35201 training step(s), loss on training batch is 0.66308.
After 35301 training step(s), loss on training batch is 0.654824.
After 35401 training step(s), loss on training batch is 0.680152.
After 35501 training step(s), loss on training batch is 0.711129.
After 35601 training step(s), loss on training batch is 0.715766.
After 35701 training step(s), loss on training batch is 0.669462.
After 35801 training step(s), loss on training batch is 0.634819.
After 35901 training step(s), loss on training batch is 0.693129.
After 36001 training step(s), loss on training batch is 0.660043.
After 36101 training step(s), loss on training batch is 0.642765.
After 36201 training step(s), loss on training batch is 0.642087.
After 36301 training step(s), loss on training batch is 0.655986.
After 36401 training step(s), loss on training batch is 0.649785.
After 36501 training step(s), loss on training batch is 0.676689.
After 36601 training step(s), loss on training batch is 0.655977.
After 36701 training step(s), loss on training batch is 0.687271.
After 36801 training step(s), loss on training batch is 0.636563.
After 36901 training step(s), loss on training batch is 0.722195.
After 37001 training step(s), loss on training batch is 0.646365.
After 37101 training step(s), loss on training batch is 0.682778.
After 37201 training step(s), loss on training batch is 0.712875.
After 37301 training step(s), loss on training batch is 0.700165.
After 37401 training step(s), loss on training batch is 0.670368.
After 37501 training step(s), loss on training batch is 0.642404.
After 37601 training step(s), loss on training batch is 0.689601.
After 37701 training step(s), loss on training batch is 0.664882.
After 37801 training step(s), loss on training batch is 0.732033.
After 37901 training step(s), loss on training batch is 0.678033.
After 38001 training step(s), loss on training batch is 0.653108.
After 38101 training step(s), loss on training batch is 0.657529.
After 38201 training step(s), loss on training batch is 0.653088.
After 38301 training step(s), loss on training batch is 0.676559.
After 38401 training step(s), loss on training batch is 0.641252.
After 38501 training step(s), loss on training batch is 0.696366.
After 38601 training step(s), loss on training batch is 0.644693.
After 38701 training step(s), loss on training batch is 0.661659.
After 38801 training step(s), loss on training batch is 0.637523.
After 38901 training step(s), loss on training batch is 0.728365.
After 39001 training step(s), loss on training batch is 0.741618.
After 39101 training step(s), loss on training batch is 0.649935.
After 39201 training step(s), loss on training batch is 0.65373.
After 39301 training step(s), loss on training batch is 0.715501.
After 39401 training step(s), loss on training batch is 0.677596.
After 39501 training step(s), loss on training batch is 0.647917.
After 39601 training step(s), loss on training batch is 0.657049.
After 39701 training step(s), loss on training batch is 0.663969.
After 39801 training step(s), loss on training batch is 0.642245.
After 39901 training step(s), loss on training batch is 0.661234.
After 40001 training step(s), loss on training batch is 0.64189.
After 40101 training step(s), loss on training batch is 0.66213.
After 40201 training step(s), loss on training batch is 0.657558.
After 40301 training step(s), loss on training batch is 0.669345.
After 40401 training step(s), loss on training batch is 0.641038.
After 40501 training step(s), loss on training batch is 0.648139.
After 40601 training step(s), loss on training batch is 0.739564.
After 40701 training step(s), loss on training batch is 0.682674.
After 40801 training step(s), loss on training batch is 0.65008.
After 40901 training step(s), loss on training batch is 0.641733.
After 41001 training step(s), loss on training batch is 0.66162.
After 41101 training step(s), loss on training batch is 0.688323.
After 41201 training step(s), loss on training batch is 0.638729.
After 41301 training step(s), loss on training batch is 0.656719.
After 41401 training step(s), loss on training batch is 0.71819.
After 41501 training step(s), loss on training batch is 0.660427.
After 41601 training step(s), loss on training batch is 0.646072.
After 41701 training step(s), loss on training batch is 0.749864.
After 41801 training step(s), loss on training batch is 0.729362.
After 41901 training step(s), loss on training batch is 0.6442.
After 42001 training step(s), loss on training batch is 0.673541.
After 42101 training step(s), loss on training batch is 0.716233.
After 42201 training step(s), loss on training batch is 0.656033.
After 42301 training step(s), loss on training batch is 0.662848.
After 42401 training step(s), loss on training batch is 0.667195.
After 42501 training step(s), loss on training batch is 0.639712.
After 42601 training step(s), loss on training batch is 0.747922.
After 42701 training step(s), loss on training batch is 0.648236.
After 42801 training step(s), loss on training batch is 0.654972.
After 42901 training step(s), loss on training batch is 0.642408.
After 43001 training step(s), loss on training batch is 0.730929.
After 43101 training step(s), loss on training batch is 0.688454.
After 43201 training step(s), loss on training batch is 0.639928.
After 43301 training step(s), loss on training batch is 0.697019.
After 43401 training step(s), loss on training batch is 0.671624.
After 43501 training step(s), loss on training batch is 0.637894.
After 43601 training step(s), loss on training batch is 0.647245.
After 43701 training step(s), loss on training batch is 0.647961.
After 43801 training step(s), loss on training batch is 0.636848.
After 43901 training step(s), loss on training batch is 0.695003.
After 44001 training step(s), loss on training batch is 0.636374.
After 44101 training step(s), loss on training batch is 0.632463.
After 44201 training step(s), loss on training batch is 0.722271.
After 44301 training step(s), loss on training batch is 0.652803.
After 44401 training step(s), loss on training batch is 0.678962.
After 44501 training step(s), loss on training batch is 0.673651.
After 44601 training step(s), loss on training batch is 0.636177.
After 44701 training step(s), loss on training batch is 0.780282.
After 44801 training step(s), loss on training batch is 0.640216.
After 44901 training step(s), loss on training batch is 0.637387.
After 45001 training step(s), loss on training batch is 0.684428.
After 45101 training step(s), loss on training batch is 0.701294.
After 45201 training step(s), loss on training batch is 0.640545.
After 45301 training step(s), loss on training batch is 0.697781.
After 45401 training step(s), loss on training batch is 0.694798.
After 45501 training step(s), loss on training batch is 0.656247.
After 45601 training step(s), loss on training batch is 0.633435.
After 45701 training step(s), loss on training batch is 0.637885.
After 45801 training step(s), loss on training batch is 0.655429.
After 45901 training step(s), loss on training batch is 0.751348.
After 46001 training step(s), loss on training batch is 0.657689.
After 46101 training step(s), loss on training batch is 0.661133.
After 46201 training step(s), loss on training batch is 0.635437.
After 46301 training step(s), loss on training batch is 0.660538.
After 46401 training step(s), loss on training batch is 0.657069.
After 46501 training step(s), loss on training batch is 0.686421.
After 46601 training step(s), loss on training batch is 0.641519.
After 46701 training step(s), loss on training batch is 0.660175.
After 46801 training step(s), loss on training batch is 0.675642.
After 46901 training step(s), loss on training batch is 0.679733.
After 47001 training step(s), loss on training batch is 0.654984.
Traceback (most recent call last):
  File "D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/7_2/mnist_lenet5_backward.py", line 82, in <module>
    main()
  File "D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/7_2/mnist_lenet5_backward.py", line 78, in main
    backward(mnist)
  File "D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/7_2/mnist_lenet5_backward.py", line 73, in backward
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
  File "C:\Python37\lib\site-packages\tensorflow\python\training\saver.py", line 1181, in save
    save_relative_paths=self._save_relative_paths)
  File "C:\Python37\lib\site-packages\tensorflow\python\training\checkpoint_management.py", line 242, in update_checkpoint_state_internal
    text_format.MessageToString(ckpt))
  File "C:\Python37\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 547, in atomic_write_string_to_file
    rename(temp_pathname, filename, overwrite)
  File "C:\Python37\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 508, in rename
    rename_v2(oldname, newname, overwrite)
  File "C:\Python37\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 526, in rename_v2
    compat.as_bytes(src), compat.as_bytes(dst), overwrite, status)
  File "C:\Python37\lib\site-packages\tensorflow\python\framework\errors_impl.py", line 528, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.UnknownError: Failed to rename: ./model\checkpoint.tmpec679952c87543f1adbd45d8f61ca058 to: ./model\checkpoint : \udcbeܾ\udcf8\udcb7\udcc3\udcceʡ\udca3
; Input/output error

Process finished with exit code 1
'''
