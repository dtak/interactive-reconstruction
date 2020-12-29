import tensorflow as tf
import numpy as np

# Replicates the architecture in Table 4 of https://arxiv.org/pdf/1802.05983.pdf

def create_recognition_network(x, K, train=False, scope='ae'):
    net = tf.reshape(x, [-1, 28, 28, 1])

    net = tf.layers.conv2d(net, 64, 4, 2, padding='SAME', use_bias=False, activation=None, name=scope+'/enc1')
    net = tf.layers.batch_normalization(net, name=scope+'/enc1_bn', training=train)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 128, 4, 2, padding='SAME', use_bias=False, activation=None, name=scope+'/enc2')
    net = tf.layers.batch_normalization(net, name=scope+'/enc2_bn', training=train)
    net = tf.nn.relu(net)

    net = tf.reshape(net, [-1, 7*7*128])

    net = tf.layers.dense(net, 1024, activation=None, name=scope+'/enc3')
    net = tf.layers.batch_normalization(net, name=scope+'/enc3_bn', training=train)
    net = tf.nn.relu(net)

    z_mean = tf.layers.dense(net, K, activation=None, name=scope+'/enc4_mean')
    z_log_sigma_sq = tf.layers.dense(net, K, activation=None, name=scope+'/enc4_log_sigma_sq')
    return z_mean, z_log_sigma_sq

def create_generator_network(z, reuse=None, train=False, scope='ae'):
    net = z

    net = tf.layers.dense(net, 1024, activation=None,  name=scope+'/dec1', reuse=reuse)
    net = tf.layers.batch_normalization(net, name=scope+'/dec1_bn', training=train, reuse=reuse)
    net = tf.nn.leaky_relu(net)

    net = tf.layers.dense(net, 7*7*128, activation=None, name=scope+'/dec2', reuse=reuse)
    net = tf.layers.batch_normalization(net, name=scope+'/dec2_bn', training=train, reuse=reuse)
    net = tf.nn.leaky_relu(net)

    net = tf.reshape(net, [-1, 7, 7, 128])

    net = tf.layers.conv2d_transpose(net, 64, 4, 2, padding='SAME', use_bias=False, activation=None, name=scope+'/dec3', reuse=reuse)
    net = tf.layers.batch_normalization(net, name=scope+'/dec3_bn', training=train, reuse=reuse)
    net = tf.nn.leaky_relu(net)

    net = tf.layers.conv2d_transpose(net, 1, 4, 2, padding='SAME', use_bias=False, activation=None, name=scope+'/dec4', reuse=reuse)
    net = tf.reshape(net, [-1, 28*28])
    return net

# Convert a numpy array of ordinal numbers to a one-hot representation

def onehot(Y, K=None):
    if K is None:
        K = np.unique(Y)
    elif isinstance(K, (int, np.int32, np.int64)):
        K = list(range(K))
    return np.array([[y == k for k in K] for y in Y]).astype(int)
