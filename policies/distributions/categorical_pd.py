import tensorflow as tf
import numpy as np
from policies.distributions.base import Distribution

class CategoricalPd(Distribution):
    """
        General methods for a diagonal gaussian distribution of this size
        """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence of two multivariate
        Gaussian distribution with diagonal covariance matrices

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """

        # assert ranks

        old_logits = old_dist_info_vars
        new_logits = new_dist_info_vars

        a0 = old_logits - tf.reduce_max(old_logits, axis=-1, keepdims=True)
        a1 = new_logits - tf.reduce_max(new_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices

       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """
        old_logits = old_dist_info
        new_logits = new_dist_info

        a0 = old_logits - np.amax(old_logits, axis=-1, keepdims=True)
        a1 = new_logits - np.amax(new_logits, axis=-1, keepdims=True)
        ea0 = np.exp(a0)
        ea1 = np.exp(a1)
        z0 = np.sum(ea0, axis=-1, keepdims=True)
        z1 = np.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return np.sum(p0 * (a0 - np.log(z0) - a1 + np.log(z1)), axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (tf.Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): likelihood ratio
        """
        # print(x_var)
        # print(new_dist_info_vars)
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)

        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, logits):
        """
        Symbolic log likelihood log p(x) of the distribution

        Args:
            x_var (tf.Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as tf.Tensor

        Returns:
             (numpy array): log likelihood
        """
        target = tf.one_hot(x_var,
                   self._dim,
                   dtype=tf.float32)

        neg_log = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=target)

        return -neg_log

    def log_likelihood(self, xs, logits):
        """
        Compute the log likelihood log p(x) of the distribution

        Args:
           x_var (numpy array): variable where to evaluate the log likelihood
           dist_info_vars (dict) : dict of distribution parameters as numpy array

        Returns:
            (numpy array): log likelihood
        """
        softmax_pd = np.exp(logits) / sum(np.exp(logits))

        targets_shape = list(np.array(xs).shape)
        final_shape = targets_shape.append(self._dim)

        targets = np.array(xs).reshape(-1)
        one_hot_targets = np.eye(self._dim)[targets].reshape(final_shape)

        log_p = np.sum(np.log(one_hot_targets *softmax_pd), axis=-1)

        return log_p

    def entropy_sym(self, logits):
        """
        Symbolic entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): entropy
        """

        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def entropy(self, logits):
        """
        Compute the entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as numpy array

        Returns:
          (numpy array): entropy
        """

        a0 = logits - np.amax(logits, axis=-1, keepdims=True)
        ea0 = np.exp(a0)
        z0 = np.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return np.sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self, logits):
        """
        Draws a sample from the distribution

        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array

        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)

        return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

    @property
    def dist_info_specs(self):
        return [("logits", (self.dim,))]