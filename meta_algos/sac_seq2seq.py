# this is the implementation of SAC using Seq2seq neural network
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class SAC():
    def __init__(self):
        pass

    def __init__(self,
                 policy,
                 q_values,
                 reply_memory,
                 lr=1e-4,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self._policy = policy
        pass

    def _build_graph(self):
        pass

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        pass

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.


        """
        policy_input_encoder = self._policy.encoder_input
        policy_input_decoder_full_length = self._policy.encoder_input
        policy_input_decoder_input = self._policy.decoder_input

        actions = self._policy.get_actions()
        log_pis = self._policy.log_pis()

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        alpha_loss = -tf.reduce_mean(
            log_alpha * tf.stop_gradient(log_pis + self._target_entropy)
        )

        self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
            self._policy_lr, name='alpha_optimizer')

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc = tf.zeros(self._action_shape),
                scale_diag= tf.ones(self._action_shape)
            )

            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        q_observations = self.q_network.encoder_input
        q_decoder_full_length = self.q_network.decoder_input

        q_log_targets = self.q_network.q_log_targets
        min_q_log_target = tf.reduce_min(q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis - min_q_log_target - policy_prior_log_probs
            )
        else:
            raise NotImplemented

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        return policy_train_op



