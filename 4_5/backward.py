#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 17:54
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
import forward
import generateds
import matplotlib.pyplot as plt
import numpy as np
# 0导入模块，生成模拟数据集
import tensorflow as tf

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = generateds.generateds()

    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print('After %d steps, loss is: %f' % (i, loss_v))

        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__ == '__main__':
    backward()

'''
C:\Python37\python.exe D:/yuangezhizao/Documents/PycharmProjects/Tensorflow-Study/4_5/backward.py
[[-4.16757847e-01 -5.62668272e-02]
 [-2.13619610e+00  1.64027081e+00]
 [-1.79343559e+00 -8.41747366e-01]
 [ 5.02881417e-01 -1.24528809e+00]
 [-1.05795222e+00 -9.09007615e-01]
 [ 5.51454045e-01  2.29220801e+00]
 [ 4.15393930e-02 -1.11792545e+00]
 [ 5.39058321e-01 -5.96159700e-01]
 [-1.91304965e-02  1.17500122e+00]
 [-7.47870949e-01  9.02525097e-03]
 [-8.78107893e-01 -1.56434170e-01]
 [ 2.56570452e-01 -9.88779049e-01]
 [-3.38821966e-01 -2.36184031e-01]
 [-6.37655012e-01 -1.18761229e+00]
 [-1.42121723e+00 -1.53495196e-01]
 [-2.69056960e-01  2.23136679e+00]
 [-2.43476758e+00  1.12726505e-01]
 [ 3.70444537e-01  1.35963386e+00]
 [ 5.01857207e-01 -8.44213704e-01]
 [ 9.76147160e-06  5.42352572e-01]
 [-3.13508197e-01  7.71011738e-01]
 [-1.86809065e+00  1.73118467e+00]
 [ 1.46767801e+00 -3.35677339e-01]
 [ 6.11340780e-01  4.79705919e-02]
 [-8.29135289e-01  8.77102184e-02]
 [ 1.00036589e+00 -3.81092518e-01]
 [-3.75669423e-01 -7.44707629e-02]
 [ 4.33496330e-01  1.27837923e+00]
 [-6.34679305e-01  5.08396243e-01]
 [ 2.16116006e-01 -1.85861239e+00]
 [-4.19316482e-01 -1.32328898e-01]
 [-3.95702397e-02  3.26003433e-01]
 [-2.04032305e+00  4.62555231e-02]
 [-6.77675577e-01 -1.43943903e+00]
 [ 5.24296430e-01  7.35279576e-01]
 [-6.53250268e-01  8.42456282e-01]
 [-3.81516482e-01  6.64890091e-02]
 [-1.09873895e+00  1.58448706e+00]
 [-2.65944946e+00 -9.14526229e-02]
 [ 6.95119605e-01 -2.03346655e+00]
 [-1.89469265e-01 -7.72186654e-02]
 [ 8.24703005e-01  1.24821292e+00]
 [-4.03892269e-01 -1.38451867e+00]
 [ 1.36723542e+00  1.21788563e+00]
 [-4.62005348e-01  3.50888494e-01]
 [ 3.81866234e-01  5.66275441e-01]
 [ 2.04207979e-01  1.40669624e+00]
 [-1.73795950e+00  1.04082395e+00]
 [ 3.80471970e-01 -2.17135269e-01]
 [ 1.17353150e+00 -2.34360319e+00]
 [ 1.16152149e+00  3.86078048e-01]
 [-1.13313327e+00  4.33092555e-01]
 [-3.04086439e-01  2.58529487e+00]
 [ 1.83533272e+00  4.40689872e-01]
 [-7.19253841e-01 -5.83414595e-01]
 [-3.25049628e-01 -5.60234506e-01]
 [-9.02246068e-01 -5.90972275e-01]
 [-2.76179492e-01 -5.16883894e-01]
 [-6.98589950e-01 -9.28891925e-01]
 [ 2.55043824e+00 -1.47317325e+00]
 [-1.02141473e+00  4.32395701e-01]
 [-3.23580070e-01  4.23824708e-01]
 [ 7.99179995e-01  1.26261366e+00]
 [ 7.51964849e-01 -9.93760983e-01]
 [ 1.10914328e+00 -1.76491773e+00]
 [-1.14421297e-01 -4.98174194e-01]
 [-1.06079904e+00  5.91666521e-01]
 [-1.83256574e-01  1.01985473e+00]
 [-1.48246548e+00  8.46311892e-01]
 [ 4.97940148e-01  1.26504175e-01]
 [-1.41881055e+00 -2.51774118e-01]
 [-1.54667461e+00 -2.08265194e+00]
 [ 3.27974540e+00  9.70861320e-01]
 [ 1.79259285e+00 -4.29013319e-01]
 [ 6.96197980e-01  6.97416272e-01]
 [ 6.01515814e-01  3.65949071e-03]
 [-2.28247558e-01 -2.06961226e+00]
 [ 6.10144086e-01  4.23496900e-01]
 [ 1.11788673e+00 -2.74242089e-01]
 [ 1.74181219e+00 -4.47500876e-01]
 [-1.25542722e+00  9.38163671e-01]
 [-4.68346260e-01 -1.25472031e+00]
 [ 1.24823646e-01  7.56502143e-01]
 [ 2.41439629e-01  4.97425649e-01]
 [ 4.10869262e+00  8.21120877e-01]
 [ 1.53176032e+00 -1.98584577e+00]
 [ 3.65053516e-01  7.74082033e-01]
 [-3.64479092e-01 -8.75979478e-01]
 [ 3.96520159e-01 -3.14617436e-01]
 [-5.93755583e-01  1.14950057e+00]
 [ 1.33556617e+00  3.02629336e-01]
 [-4.54227855e-01  5.14370717e-01]
 [ 8.29458431e-01  6.30621967e-01]
 [-1.45336435e+00 -3.38017777e-01]
 [ 3.59133332e-01  6.22220414e-01]
 [ 9.60781945e-01  7.58370347e-01]
 [-1.13431848e+00 -7.07420888e-01]
 [-1.22142917e+00  1.80447664e+00]
 [ 1.80409807e-01  5.53164274e-01]
 [ 1.03302907e+00 -3.29002435e-01]
 [-1.15100294e+00 -4.26522471e-01]
 [-1.48147191e-01  1.50143692e+00]
 [ 8.69598198e-01 -1.08709057e+00]
 [ 6.64221413e-01  7.34884668e-01]
 [-1.06136574e+00 -1.08516824e-01]
 [-1.85040397e+00  3.30488064e-01]
 [-3.15693210e-01 -1.35000210e+00]
 [-6.98170998e-01  2.39951198e-01]
 [-5.52949440e-01  2.99526813e-01]
 [ 5.52663696e-01 -8.40443012e-01]
 [-3.12270670e-01  2.14467809e+00]
 [ 1.21105582e-01 -8.46828752e-01]
 [ 6.04624490e-02 -1.33858888e+00]
 [ 1.13274608e+00  3.70304843e-01]
 [ 1.08580640e+00  9.02179395e-01]
 [ 3.90296450e-01  9.75509412e-01]
 [ 1.91573647e-01 -6.62209012e-01]
 [-1.02351498e+00 -4.48174823e-01]
 [-2.50545813e+00  1.82599446e+00]
 [-1.71406741e+00 -7.66395640e-02]
 [-1.31756727e+00 -2.02559359e+00]
 [-8.22453750e-02 -3.04666585e-01]
 [-1.59724130e-01  5.48946560e-01]
 [-6.18375485e-01  3.78794466e-01]
 [ 5.13251444e-01 -3.34844125e-01]
 [-2.83519516e-01  5.38424263e-01]
 [ 5.72509465e-02  1.59088487e-01]
 [-2.37440268e+00  5.85199353e-02]
 [ 3.76545911e-01 -1.35479764e-01]
 [ 3.35908395e-01  1.90437591e+00]
 [ 8.53644334e-02  6.65334278e-01]
 [-8.49995503e-01 -8.52341797e-01]
 [-4.79985112e-01 -1.01964910e+00]
 [-7.60113841e-03 -9.33830661e-01]
 [-1.74996844e-01 -1.43714343e+00]
 [-1.65220029e+00 -6.75661789e-01]
 [-1.06706712e+00 -6.52931145e-01]
 [-6.12094750e-01 -3.51262461e-01]
 [ 1.04547799e+00  1.36901602e+00]
 [ 7.25353259e-01 -3.59474459e-01]
 [ 1.49695179e+00 -1.53111111e+00]
 [-2.02336394e+00  2.67972576e-01]
 [-2.20644541e-03 -1.39291883e-01]
 [ 3.25654693e-02 -1.64056022e+00]
 [-1.15669917e+00  1.23403468e+00]
 [ 1.02818490e+00 -7.21879726e-01]
 [ 1.93315697e+00 -1.07079633e+00]
 [-5.71381608e-01  2.92432067e-01]
 [-1.19499989e+00 -4.87930544e-01]
 [-1.73071165e-01 -3.95346401e-01]
 [ 8.70840765e-01  5.92806797e-01]
 [-1.09929731e+00 -6.81530644e-01]
 [ 1.80066685e-01 -6.69310440e-02]
 [-7.87749540e-01  4.24753672e-01]
 [ 8.19885117e-01 -6.31118683e-01]
 [ 7.89059649e-01 -1.62167380e+00]
 [-1.61049926e+00  4.99939764e-01]
 [-8.34515207e-01 -9.96959687e-01]
 [-2.63388077e-01 -6.77360492e-01]
 [ 3.27067038e-01 -1.45535944e+00]
 [-3.71519124e-01  3.16096597e+00]
 [ 1.09951013e-01 -1.91352322e+00]
 [ 5.99820429e-01  5.49384465e-01]
 [ 1.38378103e+00  1.48349243e-01]
 [-6.53541444e-01  1.40883398e+00]
 [ 7.12061227e-01 -1.80071604e+00]
 [ 7.47598942e-01 -2.32897001e-01]
 [ 1.11064528e+00 -3.73338813e-01]
 [ 7.86146070e-01  1.94168696e-01]
 [ 5.86204098e-01 -2.03872918e-02]
 [-4.14408598e-01  6.73134124e-02]
 [ 6.31798924e-01  4.17592731e-01]
 [ 1.61517627e+00  4.25606211e-01]
 [ 6.35363758e-01  2.10222927e+00]
 [ 6.61264168e-02  5.35558351e-01]
 [-6.03140792e-01  4.19576292e-02]
 [ 1.64191464e+00  3.11697707e-01]
 [ 1.45116990e+00 -1.06492788e+00]
 [-1.40084545e+00  3.07525527e-01]
 [-1.36963867e+00  2.67033724e+00]
 [ 1.24845030e+00 -1.24572655e+00]
 [-1.67168774e-01 -5.76610930e-01]
 [ 4.16021749e-01 -5.78472626e-02]
 [ 9.31887358e-01  1.46833213e+00]
 [-2.21320943e-01 -1.17315562e+00]
 [ 5.62669078e-01 -1.64515057e-01]
 [ 1.14485538e+00 -1.52117687e-01]
 [ 8.29789046e-01  3.36065952e-01]
 [-1.89044051e-01 -4.49328601e-01]
 [ 7.13524448e-01  2.52973487e+00]
 [ 8.37615794e-01 -1.31682403e-01]
 [ 7.07592866e-01  1.14053878e-01]
 [-1.28089518e+00  3.09846277e-01]
 [ 1.54829069e+00 -3.15828043e-01]
 [-1.12590378e+00  4.88496666e-01]
 [ 1.83094666e+00  9.40175993e-01]
 [ 1.01871705e+00  2.30237829e+00]
 [ 1.62109298e+00  7.12683273e-01]
 [-2.08703629e-01  1.37617991e-01]
 [-1.03352168e-01  8.48350567e-01]
 [-8.83125561e-01  1.54538683e+00]
 [ 1.45840073e-01 -4.00106056e-01]
 [ 8.15206041e-01 -2.07492237e+00]
 [-8.34437391e-01 -6.57718447e-01]
 [ 8.20564332e-01 -4.89157001e-01]
 [ 1.42496703e+00 -4.46857897e-01]
 [ 5.21109431e-01 -7.08194380e-01]
 [ 1.15553059e+00 -2.54530459e-01]
 [ 5.18924924e-01 -4.92994911e-01]
 [-1.08654815e+00 -2.30917497e-01]
 [ 1.09801004e+00 -1.01787805e+00]
 [-1.52939136e+00 -3.07987737e-01]
 [ 7.80754356e-01 -1.05583964e+00]
 [-5.43883381e-01  1.84301739e-01]
 [-3.30675843e-01  2.87208202e-01]
 [ 1.18952814e+00  2.12015479e-02]
 [-6.54096803e-02  7.66115904e-01]
 [-6.16350846e-02 -9.52897152e-01]
 [-1.01446306e+00 -1.11526396e+00]
 [ 1.91260068e+00 -4.52632031e-02]
 [ 5.76909718e-01  7.17805695e-01]
 [-9.38998998e-01  6.28775807e-01]
 [-5.64493432e-01 -2.08780746e+00]
 [-2.15050132e-01 -1.07502856e+00]
 [-3.37972149e-01  3.43212732e-01]
 [ 2.28253964e+00 -4.95778848e-01]
 [-1.63962832e-01  3.71622161e-01]
 [ 1.86521520e-01 -1.58429224e-01]
 [-1.08292956e+00 -9.56625520e-01]
 [-1.83376735e-01 -1.15980690e+00]
 [-6.57768362e-01 -1.25144841e+00]
 [ 1.12448286e+00 -1.49783981e+00]
 [ 1.90201722e+00 -5.80383038e-01]
 [-1.05491567e+00 -1.18275720e+00]
 [ 7.79480054e-01  1.02659795e+00]
 [-8.48666001e-01  3.31539648e-01]
 [-1.49591353e-01 -2.42440600e-01]
 [ 1.51197175e-01  7.65069481e-01]
 [-1.91663052e+00 -2.22734129e+00]
 [ 2.06689897e-01 -7.08763560e-02]
 [ 6.84759969e-01 -1.70753905e+00]
 [-9.86569665e-01  1.54353634e+00]
 [-1.31027053e+00  3.63433972e-01]
 [-7.94872445e-01 -4.05286267e-01]
 [-1.37775793e+00  1.18604868e+00]
 [-1.90382114e+00 -1.19814038e+00]
 [-9.10065643e-01  1.17645419e+00]
 [ 2.99210670e-01  6.79267178e-01]
 [-1.76606800e-02  2.36040923e-01]
 [ 4.94035871e-01  1.54627765e+00]
 [ 2.46857508e-01 -1.46877580e+00]
 [ 1.14709994e+00  9.55569845e-02]
 [-1.10743873e+00 -1.76286141e-01]
 [-9.82755667e-01  2.08668273e+00]
 [-3.44623671e-01 -2.00207923e+00]
 [ 3.03234433e-01 -8.29874845e-01]
 [ 1.28876941e+00  1.34925462e-01]
 [-1.77860064e+00 -5.00791490e-01]
 [-1.08816157e+00 -7.57855553e-01]
 [-6.43744900e-01 -2.00878453e+00]
 [ 1.96262894e-01 -8.75896370e-01]
 [-8.93609209e-01  7.51902355e-01]
 [ 1.89693224e+00 -6.29079151e-01]
 [ 1.81208553e+00 -2.05626574e+00]
 [ 5.62704887e-01 -5.82070757e-01]
 [-7.40029749e-02 -9.86496364e-01]
 [-5.94722499e-01 -3.14811843e-01]
 [-3.46940532e-01  4.11443516e-01]
 [ 2.32639090e+00 -6.34053128e-01]
 [-1.54409962e-01 -1.74928880e+00]
 [-2.51957930e+00  1.39116243e+00]
 [-1.32934644e+00 -7.45596414e-01]
 [ 2.12608498e-02  9.10917515e-01]
 [ 3.15276082e-01  1.86620821e+00]
 [-1.82497623e-01 -1.82826634e+00]
 [ 1.38955717e-01  1.19450165e-01]
 [-8.18899200e-01 -3.32639265e-01]
 [-5.86387955e-01  1.73451634e+00]
 [-6.12751558e-01 -1.39344202e+00]
 [ 2.79433757e-01 -1.82223127e+00]
 [ 4.27017458e-01  4.06987749e-01]
 [-8.44308241e-01 -5.59820113e-01]
 [-6.00520405e-01  1.61487324e+00]
 [ 3.94953220e-01 -1.20381347e+00]
 [-1.24747243e+00 -7.75462496e-02]
 [-1.33397514e-02 -7.68323250e-01]
 [ 2.91234010e-01 -1.97330948e-01]
 [ 1.07682965e+00  4.37410232e-01]
 [-9.31978663e-02  1.35631416e-01]
 [-8.82708822e-01  8.84744194e-01]
 [ 3.83204463e-01 -4.16994149e-01]
 [ 1.17796550e-01 -5.36685309e-01]
 [ 2.48718458e+00 -4.51361054e-01]
 [ 5.18836127e-01  3.64448005e-01]
 [-7.98348729e-01  5.65779713e-03]
 [-3.20934708e-01  2.49513550e-01]
 [ 2.56308392e-01  7.67625083e-01]
 [ 7.83020087e-01 -4.07063047e-01]
 [-5.24891667e-01 -5.89808683e-01]
 [-8.62531086e-01 -1.74287290e+00]]
[[1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [0]
 [0]
 [1]
 [0]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [0]
 [0]
 [0]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]]
[['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['blue'], ['red'], ['blue'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['blue'], ['red'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['blue'], ['blue'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue'], ['red'], ['red'], ['red'], ['red'], ['red'], ['red'], ['blue']]
WARNING:tensorflow:From C:\Python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.

WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
If you depend on functionality not listed there, please file an issue.

2019-03-23 19:45:24.083908: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-03-23 19:45:24.548956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 0.9495
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.63GiB
2019-03-23 19:45:24.555279: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-23 19:45:25.154259: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-23 19:45:25.158078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-23 19:45:25.160148: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-23 19:45:25.162389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 716 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-03-23 19:45:25.343309: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library cublas64_100.dll locally
After 0 steps, loss is: 4.905663
After 2000 steps, loss is: 0.179908
After 4000 steps, loss is: 0.138143
After 6000 steps, loss is: 0.113342
After 8000 steps, loss is: 0.100654
After 10000 steps, loss is: 0.093703
After 12000 steps, loss is: 0.090163
After 14000 steps, loss is: 0.089534
After 16000 steps, loss is: 0.089487
After 18000 steps, loss is: 0.089476
After 20000 steps, loss is: 0.089459
After 22000 steps, loss is: 0.089429
After 24000 steps, loss is: 0.089398
After 26000 steps, loss is: 0.089371
After 28000 steps, loss is: 0.089347
After 30000 steps, loss is: 0.089325
After 32000 steps, loss is: 0.089306
After 34000 steps, loss is: 0.089289
After 36000 steps, loss is: 0.089277
After 38000 steps, loss is: 0.089267

Process finished with exit code 0
'''