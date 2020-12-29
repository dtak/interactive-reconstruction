import numpy as np
import tensorflow as tf

# Code adapted from https://github.com/miyosuda/disentangled_vae;
# Replicates the architecture in https://arxiv.org/pdf/1804.03599.pdf

def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

def _conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                      padding='SAME')

def _conv2d_weight_variable(weight_shape, name, deconv=False):
  name_w = "W_{0}".format(name)
  name_b = "b_{0}".format(name)
  w = weight_shape[0]
  h = weight_shape[1]
  if deconv:
    input_channels  = weight_shape[3]
    output_channels = weight_shape[2]
  else:
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
  d = 1.0 / np.sqrt(input_channels * w * h)
  bias_shape = [output_channels]
  weight = tf.get_variable(name_w, weight_shape,
                           initializer=conv_initializer(w, h, input_channels))
  bias   = tf.get_variable(name_b, bias_shape,
                           initializer=conv_initializer(w, h, input_channels))
  return weight, bias

def _fc_weight_variable(weight_shape, name):
  name_w = "W_{0}".format(name)
  name_b = "b_{0}".format(name)
  input_channels  = weight_shape[0]
  output_channels = weight_shape[1]
  d = 1.0 / np.sqrt(input_channels)
  bias_shape = [output_channels]
  weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
  bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
  return weight, bias

def _get_deconv2d_output_size(input_height, input_width, filter_height,
                              filter_width, row_stride, col_stride, padding_type):
  if padding_type == 'VALID':
    out_height = (input_height - 1) * row_stride + filter_height
    out_width  = (input_width  - 1) * col_stride + filter_width
  elif padding_type == 'SAME':
    out_height = input_height * row_stride
    out_width  = input_width * col_stride
  return out_height, out_width

def _deconv2d(x, W, input_width, input_height, stride):
  filter_height = W.get_shape()[0].value
  filter_width  = W.get_shape()[1].value
  out_channel   = W.get_shape()[2].value
  
  out_height, out_width = _get_deconv2d_output_size(input_height,
                                                         input_width,
                                                         filter_height,
                                                         filter_width,
                                                         stride,
                                                         stride,
                                                         'SAME')
  batch_size = tf.shape(x)[0]
  output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
  return tf.nn.conv2d_transpose(x, W, output_shape,
                                strides=[1, stride, stride, 1],
                                padding='SAME')

def create_recognition_network(x, reuse=tf.AUTO_REUSE, variational=False, K=5, pre=False, prefix='ae', nonlinearity=tf.nn.relu):
  with tf.variable_scope(prefix+"/encoder", reuse=reuse) as scope:
    # [filter_height, filter_width, in_channels, out_channels]
    W_conv1, b_conv1 = _conv2d_weight_variable([4, 4, 1,  32], "conv1")
    W_conv2, b_conv2 = _conv2d_weight_variable([4, 4, 32, 32], "conv2")
    W_conv3, b_conv3 = _conv2d_weight_variable([4, 4, 32, 32], "conv3")
    W_conv4, b_conv4 = _conv2d_weight_variable([4, 4, 32, 32], "conv4")
    W_fc1, b_fc1     = _fc_weight_variable([4*4*32, 256], "fc1")
    W_fc2, b_fc2     = _fc_weight_variable([256, 256], "fc2")
    if not pre:
      W_fc3, b_fc3     = _fc_weight_variable([256, K],  "fc3")

    x_reshaped = tf.reshape(x, [-1, 64, 64, 1])
    h_conv1 = nonlinearity(_conv2d(x_reshaped, W_conv1, 2) + b_conv1) # (32, 32)
    h_conv2 = nonlinearity(_conv2d(h_conv1,    W_conv2, 2) + b_conv2) # (16, 16)
    h_conv3 = nonlinearity(_conv2d(h_conv2,    W_conv3, 2) + b_conv3) # (8, 8)
    h_conv4 = nonlinearity(_conv2d(h_conv3,    W_conv4, 2) + b_conv4) # (4, 4)
    h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*32])
    h_fc1 = nonlinearity(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    h_fc2 = nonlinearity(tf.matmul(h_fc1,        W_fc2) + b_fc2)
    if pre: return h_fc2
    z_mean         = tf.matmul(h_fc2, W_fc3) + b_fc3
    if variational:
      W_fc4, b_fc4     = _fc_weight_variable([256, K],  "fc4")
      z_log_sigma_sq = tf.matmul(h_fc2, W_fc4) + b_fc4
      return (z_mean, z_log_sigma_sq)
    else:
      return z_mean

def create_generator_network(z, reuse=tf.AUTO_REUSE, K=5, prefix='ae'):
  with tf.variable_scope(prefix+"/decoder", reuse=reuse) as scope:
    W_fc1, b_fc1 = _fc_weight_variable([K,  256],    "fc1")
    W_fc2, b_fc2 = _fc_weight_variable([256, 4*4*32], "fc2")

    # [filter_height, filter_width, output_channels, in_channels]
    W_deconv1, b_deconv1 = _conv2d_weight_variable([4, 4, 32, 32], "deconv1", deconv=True)
    W_deconv2, b_deconv2 = _conv2d_weight_variable([4, 4, 32, 32], "deconv2", deconv=True)
    W_deconv3, b_deconv3 = _conv2d_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
    W_deconv4, b_deconv4 = _conv2d_weight_variable([4, 4,  1, 32], "deconv4", deconv=True)

    h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_reshaped = tf.reshape(h_fc2, [-1, 4, 4, 32])
    h_deconv1   = tf.nn.relu(_deconv2d(h_fc2_reshaped, W_deconv1,  4,  4, 2) + b_deconv1)
    h_deconv2   = tf.nn.relu(_deconv2d(h_deconv1,      W_deconv2,  8,  8, 2) + b_deconv2)
    h_deconv3   = tf.nn.relu(_deconv2d(h_deconv2,      W_deconv3, 16, 16, 2) + b_deconv3)
    h_deconv4   =            _deconv2d(h_deconv3,      W_deconv4, 32, 32, 2) + b_deconv4
    
    x_out_logit = tf.reshape(h_deconv4, [-1, 64*64*1])
    return x_out_logit
