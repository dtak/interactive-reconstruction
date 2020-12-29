import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from dsprites_helpers import create_generator_network

"""
This script trains the "ground-truth" dSprites model.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--convert_to_tensorflowjs', type=int, default=1)
FLAGS = parser.parse_args()

# Prepare the directory for saving outputs
path = FLAGS.output_dir
os.system('mkdir -p ' + path)

# Load the dataset
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
data_path = os.path.join(data_dir, 'dsprites.npz')

if not os.path.exists(data_path):
    raise RuntimeError("Before running this file, download the dSprites npz file from https://github.com/deepmind/dsprites-dataset and save to data/dsprites.npz")

with np.load(data_path, encoding='bytes', allow_pickle=True) as data:
    X = data['imgs'].reshape(-1, 64*64).astype(float)
    Z = data['latents_values'][:,1:]

order = np.arange(len(X))
rs = np.random.RandomState(seed=0)
rs.shuffle(order)
n_test = int(0.1 * len(X))
trn = order[n_test:]
tst = order[:n_test]

# Define our model in Tensorflow
D = 64*64
K = 5

Z_in = tf.placeholder("float", [None, K], name='Z_true')
X_true = tf.placeholder("float", [None, D], name='X_true')
X_pred_logits = create_generator_network(Z_in)
X_pred = tf.nn.sigmoid(X_pred_logits, name='X_out')

# Define our loss function
losses = OrderedDict()

mses = tf.reduce_sum((X_true - X_pred)**2, axis=1)
nlls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_true, logits=X_pred_logits), axis=1)

losses['NLL'] = tf.reduce_mean(nlls)

loss = 0
for k,v in losses.items():
    loss += v

# Define our optimizer
lr = tf.placeholder('float', shape=())
optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initialize Tensorflow variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Define helper for running SGD
def minibatch_indexes(lenX=len(trn), batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs):
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
    feed[lr] = 0.0005
    feed[X_true] = X[idx]
    feed[Z_in] = Z[idx]
    outputs = sess.run([optim] + list(losses.values()), feed_dict=feed)
    if i % FLAGS.print_every == 0:
        parts = ['Epoch '+str(epoch+1)+', iter '+str(i)]
        parts += ['{} {:.8f}'.format(k, outputs[j+1]) for j,k in enumerate(losses.keys())]
        print(', '.join(parts))

# Save the model to disk
graph = tf.get_default_graph()
decoder = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['X_out'])
with tf.gfile.GFile(path+'/decoder.pb', "wb") as f:
    f.write(decoder.SerializeToString())

# Compute and save test metrics
test_nlls = []
test_mses = []
i = 0
while i < len(tst):
    feed = {}
    feed[X_true] = X[tst][i:i+2000]
    feed[Z_in] = Z[tst][i:i+2000]
    test_nlls.append(sess.run(nlls, feed_dict=feed))
    test_mses.append(sess.run(mses, feed_dict=feed))
    i += 2000
test_nlls = np.hstack(test_nlls)
test_mses = np.hstack(test_mses)

with open(path+'/nll.txt', 'w') as f:
    f.write(str(np.mean(test_nlls)))

with open(path+'/mse.txt', 'w') as f:
    f.write(str(np.mean(test_mses)))

# Save some extra configuration about the model in a JSON file that the web
# interface will use to properly render inputs
Z_pcts = {}
for pct in [0,1,2,5,25,33,45,48,50,52,55,67,75,95,98,99,100]:
    Z_pcts[pct] = [np.percentile(Z[:,i], pct) for i in range(K)]

config = {
    'vars': [{ 'type': 'continuous' } for _ in range(K)],
    'Dx': D,
    'Dc': [],
    'Dz': K,
    'z': Z[tst][:250]
    'c': [],
    'z_pcts': Z_pcts,
    'z_lims': [[np.min(Z[:,i]),
                np.max(Z[:,i])] for i in range(K)]
}

with open(path+'/config.json', 'w') as f:
    f.write(json.dumps(config, indent=4, cls=NumpyEncoder))

# Convert the model to TensorflowJS format
if FLAGS.convert_to_tensorflowjs:
    os.system("tensorflowjs_converter"
              + " --input_format=tf_frozen_model"
              + " --output_node_names=X_out"
              + f" {path}/decoder.pb {path}/decoder_web")
    rename_tensorflowjs_manifests(f"{path}/decoder_web/weights_manifest.json")
