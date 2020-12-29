import math
import numpy as np
import tensorflow as tf
import scipy
import scipy.stats
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Helpers for running beta-TCVAE
# References:
# - https://arxiv.org/abs/1802.04942
# - https://github.com/julian-carpenter/beta-TCVAE/blob/572d9e31993ccce47ef7a072a49c027c9c944e5e/nn/losses.py#L93
def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

def total_correlation(z, z_mean, z_logvar):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)

# Helpers for estimating the mutual information gap
# References:
# - https://arxiv.org/abs/1802.04942
def estimate_mutual_information(X, Y, bins=20):
  hist = np.histogram2d(X, Y, bins)[0] # approximate joint
  info = mutual_info_score(None, None, contingency=hist)
  return info / np.log(2) # bits

def estimate_entropy(X, **kw):
  return estimate_mutual_information(X, X, **kw)

def estimate_mutual_information_gap(Z_true, Z_learned, **kw):
  K = Z_true.shape[1]
  gap = 0
  for k in range(K):
    H = estimate_entropy(Z_true[:,k], **kw)
    MIs = sorted([
      estimate_mutual_information(Z_learned[:,j], Z_true[:,k], **kw)
      for j in range(Z_learned.shape[1])
    ], reverse=True)
    gap += (MIs[0] - MIs[1]) / (H * K)
  return gap

# Helpers for estimating the DCI score
# References:
# - https://openreview.net/forum?id=By-7dz-AZ
# - https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/dci.py
def estimate_DCI_scores(gen_factors, latents):
  """Computes score based on both training and testing codes and factors."""
  mus_train, mus_test, ys_train, ys_test = train_test_split(gen_factors, latents, test_size=0.1)
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[1]
  assert importance_matrix.shape[1] == ys_train.shape[1]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = dci_disentanglement(importance_matrix)
  scores["completeness"] = dci_completeness(importance_matrix)
  return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[1]
  num_codes = x_train.shape[1]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train[:,i])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(model.score(x_train, y_train[:,i]))
    test_loss.append(model.score(x_test, y_test[:,i]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def dci_disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)

def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def dci_completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
