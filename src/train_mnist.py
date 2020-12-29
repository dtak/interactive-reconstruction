import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from mnist_helpers import *
from disentanglement_helpers import *

"""
To replicate the experiments from the paper, you can train new versions of each
model by running this script with the following parameters:

  AE_5: --K=5 --variational=0
 VAE_5: --K=5 --variational=1 --kl_penalty=1
  TC_5: --K=5 --variational=1 --kl_penalty=1 --tc_penalty=9
  IG_5: (see below)
  SS_5: --K=5 --variational=1 --kl_penalty=1 --tc_penalty=9 --semi_supervised=1
 AE_10: --K=10 --variational=0
 TC_10: --K=10 --variational=1 --kl_penalty=1 --tc_penalty=9
 SS_10: --K=10 --variational=1 --kl_penalty=1 --tc_penalty=9 --semi_supervised=1

To train the InfoGAN, see
https://github.com/dtak/tensorpack/commit/929f1c819fb1943a72436d9958b2f19d96c5e6a5
"""

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--variational', type=int, default=1)
parser.add_argument('--kl_penalty', type=float, default=1.0)
parser.add_argument('--tc_penalty', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--convert_to_tensorflowjs', type=int, default=1)
FLAGS = parser.parse_args()

# Prepare the directory for saving outputs
path = FLAGS.output_dir
os.system('mkdir -p ' + path)

# Load the dataset
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
train, test = tf.keras.datasets.mnist.load_data(path=os.path.join(data_dir, 'mnist.npz'))
X_train, y_train = train
X_test, y_test = test
X_train = X_train.reshape(-1, 28*28) / 255.
X_test = X_test.reshape(-1, 28*28) / 255.
y_train = onehot(y_train)
y_test = onehot(y_test)

# Define our model in Tensorflow
D = 28*28
K = FLAGS.K

X_in = tf.placeholder("float", [None, D], name='X_in')
training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=())
if FLAGS.semi_supervised:
    y_in = tf.placeholder("float", [None, 10], name='y_in')
    Z_in = tf.placeholder("float", [None, K+10], name='Z_in')
else:
    Z_in = tf.placeholder("float", [None, K], name='Z_in')

Z_mean, Z_lvar = create_recognition_network(X_in, K, train=training)

if FLAGS.variational:
    eps = tf.random_normal(tf.shape(Z_mean), 0, 1, dtype=tf.float32)
    Z_out = tf.add(Z_mean, tf.multiply(tf.sqrt(tf.exp(Z_lvar)), eps))
else:
    Z_out = Z_mean

if FLAGS.semi_supervised:
    Z_y = tf.concat([y_in, Z_out], axis=1)
    X_rec_logits = create_generator_network(Z_y, train=training)
else:
    X_rec_logits = create_generator_network(Z_out, train=training)

X_rec = tf.nn.sigmoid(X_rec_logits)

X_out_logits = create_generator_network(Z_in, reuse=True, train=False)
X_out = tf.nn.sigmoid(X_out_logits, name='X_out')

# Define our loss function (separated out into separate components for easy
# logging)
losses = OrderedDict()
mses = tf.reduce_sum((X_in - X_rec)**2, axis=1)
nlls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_in, logits=X_rec_logits), axis=1)

losses['NLL'] = tf.reduce_mean(nlls)

if FLAGS.variational:
    if FLAGS.kl_penalty > 0:
        kls = -0.5 * tf.reduce_sum(1 + Z_lvar
                                     - tf.square(Z_mean)
                                     - tf.exp(Z_lvar), 1)
        losses['KL'] = FLAGS.kl_penalty * tf.reduce_mean(kls)

    if FLAGS.tc_penalty > 0:
        losses['TC'] = FLAGS.tc_penalty * total_correlation(Z_out, Z_mean, Z_lvar)

loss = 0
for k,v in losses.items():
    loss += v

# Define our optimizer
lr = tf.placeholder('float', shape=())
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
optim = tf.group([optim, update_ops])

# Initialize Tensorflow variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Define helper for running SGD
def minibatch_indexes(lenX=len(X_train), batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
  trn = np.arange(lenX)
  n = int(np.ceil(lenX / batch_size))
  for epoch in range(num_epochs):
    np.random.shuffle(trn)
    for batch in range(n):
      i = epoch*n + batch
      sl = slice((i%n)*batch_size, ((i%n)+1)*batch_size)
      yield i, epoch, trn[sl]

# Train the model
for i, epoch, idx in minibatch_indexes():
    feed = {}
    feed[X_in] = X_train[idx]
    feed[training] = True
    feed[lr] = 0.001
    if FLAGS.semi_supervised:
        feed[y_in] = y_train[idx]

    outputs = sess.run([optim] + list(losses.values()), feed_dict=feed)

    if i % FLAGS.print_every == 0:
        parts = ['Epoch '+str(epoch+1)+', iter '+str(i)]
        parts += ['{} {:.8f}'.format(k, outputs[j+1]) for j,k in enumerate(losses.keys())]
        print(', '.join(parts))

# Save the model to disk
graph = tf.get_default_graph()
decoder = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['X_out'])
with tf.gfile.GFile(path+'/decoder.pb', 'wb') as f:
    f.write(decoder.SerializeToString())

# Compute and save test metrics
Z_test = []
test_nlls = []
test_mses = []
i = 0
while i < len(X_test):
    feed = {}
    feed[X_in] = X_test[i:i+2000]
    if FLAGS.semi_supervised:
        feed[y_in] = y_test[i:i+2000]
    Z_test.append(sess.run(Z_mean, feed_dict=feed))
    test_nlls.append(sess.run(nlls, feed_dict=feed))
    test_mses.append(sess.run(mses, feed_dict=feed))
    i += 2000
Z_test = np.vstack(Z_test)
test_nlls = np.hstack(test_nlls)
test_mses = np.hstack(test_mses)

with open(path+'/nll.txt', 'w') as f:
    f.write(str(np.mean(test_nlls)))

with open(path+'/mse.txt', 'w') as f:
    f.write(str(np.mean(test_mses)))

# Save some extra configuration about the model in a JSON file that the web
# interface will use to properly render inputs
config = {
    'vars': [{ 'type': 'continuous' } for _ in range(K)],
    'Dx': D,
    'Dz': K,
    'Dc': [],
    'c': [],
    'z': Z_test[:250],
    'z_lims': [[np.min(Z_test[:,i]),
                np.max(Z_test[:,i])] for i in range(K)]
}

if FLAGS.semi_supervised:
    config['vars'] = [{'type': 'categorical', 'K': 10 }] + config['vars']
    config['Dc'] = [10]
    config['c'] = [y_test[:250]]

with open(path+'/config.json', 'w') as f:
    f.write(json.dumps(config, indent=4, cls=NumpyEncoder))

# Convert the model to TensorflowJS format
if FLAGS.convert_to_tensorflowjs:
    os.system("tensorflowjs_converter"
              + " --input_format=tf_frozen_model"
              + " --output_node_names=X_out"
              + f" {path}/decoder.pb {path}/decoder_web")
    rename_tensorflowjs_manifests(f"{path}/decoder_web/weights_manifest.json")
