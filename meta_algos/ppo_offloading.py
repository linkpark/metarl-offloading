# the eager version.

import tensorflow as tf
import numpy as np
from utils.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from policies.meta_seq2seq_policy import Seq2SeqPolicy
import itertools

class PPO():
    """
    Provides ppo based offloading training
    """
    def __init__(self,
                 policy,
                 meta_sampler,
                 meta_sampler_process,
                 lr=1e-4,
                 num_inner_grad_steps=4,
                 clip_value = 0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.lr = lr
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process

        #self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
        #self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.build_graph()

    def build_graph(self):
        new_logits = self.policy.network.decoder_logits
        self.decoder_inputs = self.policy.decoder_inputs
        self.old_logits = tf.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim])
        self.actions = self.policy.decoder_targets
        self.obs = self.policy.obs
        self.vpred = self.policy.vf
        self.decoder_full_length = self.policy.decoder_full_length

        self.old_v = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.advs = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, None])

        with tf.compat.v1.variable_scope("ppo_update") as scope:
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(self.actions, self.old_logits, new_logits)

            clipped_obj = tf.minimum(likelihood_ratio * self.advs ,
                                     tf.clip_by_value(likelihood_ratio,
                                                      1.0 - self.clip_value,
                                                      1.0 + self.clip_value) * self.advs)
            self.surr_obj = -tf.reduce_mean(clipped_obj)

            vpredclipped = self.vpred + tf.clip_by_value(self.vpred - self.old_v, -self.clip_value, self.clip_value)
            vf_losses1 = tf.square(self.vpred - self.r)
            vf_losses2 = tf.square(vpredclipped - self.r)

            self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            self.total_loss = self.surr_obj + self.vf_coef * self.vf_loss

            params = self.policy.network.get_trainable_variables()

            grads_and_var = self.optimizer.compute_gradients(self.total_loss, params)

            grads, var = zip(*grads_and_var)

            if self.max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))

            self._train = self.optimizer.apply_gradients(grads_and_var)


    def UpdatePPOTarget(self, task_samples, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)

        #observations = task_samples['observations']

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = np.split(np.array(task_samples['observations']), batch_number)
        actions_batchs = np.split(np.array(task_samples['actions']), batch_number)
        shift_action_batchs = np.split(np.array(shift_actions), batch_number)

        old_logits_batchs = np.split(np.array(task_samples["logits"], dtype=np.float32 ), batch_number)
        advs_batchs = np.split(np.array(task_samples['advantages'], dtype=np.float32), batch_number)
        oldvpred = np.split(np.array(task_samples['values'], dtype=np.float32), batch_number)
        returns = np.split(np.array(task_samples['returns'], dtype=np.float32), batch_number)

        sess = tf.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0
        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(self.num_inner_grad_steps):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {self.old_logits: old_logits, self.old_v: old_v, self.obs: observations, self.actions: actions,
                            self.decoder_inputs: shift_actions, self.decoder_full_length: decoder_full_length, self.advs: advs, self.r: r}

                _, value_loss, policy_loss = sess.run([self._train, self.vf_loss, self.surr_obj], feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses


