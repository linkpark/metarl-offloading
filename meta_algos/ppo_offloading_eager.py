import tensorflow as tf
import numpy as np
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

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)

        print("batch_number is: ", batch_number)

        #observations = task_samples['observations']

        shift_actions = np.column_stack(
                    (np.zeros(task_samples['actions'].shape[0], dtype=np.int32), task_samples['actions'][:, 0:-1]))

        observations_batchs = tf.split(tf.convert_to_tensor(task_samples['observations']), batch_number)
        actions_batchs = tf.split(tf.convert_to_tensor(task_samples['actions']), batch_number)
        shift_action_batchs = tf.split(tf.convert_to_tensor(shift_actions), batch_number)
        advs_batchs = tf.split(tf.convert_to_tensor(task_samples['advantages'], dtype=tf.float32), batch_number)
        oldvpred = tf.split(tf.convert_to_tensor(task_samples['values'], dtype=tf.float32), batch_number)
        returns = tf.split(tf.convert_to_tensor(task_samples['returns'], dtype=tf.float32), batch_number)

        # copy_policy =  Seq2SeqPolicy(self.policy.obs_dim, self.policy.encoder_units,
        #                              self.policy.decoder_units, self.policy.action_dim, value_network_dimension=1)
        #
        # copy_policy(observations)

        # inner loss for ppo target:
        policy_loss = 0.0
        value_loss = 0.0

        old_logits_batchs = []
        for observations, shift_actions in zip(observations_batchs, shift_action_batchs):
            old_logits, _ = self.policy(observations, shift_actions)
            old_logits_batchs.append(old_logits)

        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(self.num_inner_grad_steps):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                with tf.GradientTape() as inner_tape:
                    new_logits, vpred = self.policy(observations, shift_actions)

                    likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(actions, old_logits, new_logits)

                    #print("likelihhod_ratio is: ", likelihood_ratio[0].numpy())

                    clipped_obj = tf.minimum(likelihood_ratio * advs,
                                             tf.clip_by_value(likelihood_ratio,
                                                              1.0 - self.clip_value,
                                                              1.0 + self.clip_value) * advs)

                    surr_obj = -tf.reduce_mean(clipped_obj)

                    vpredclipped = vpred + tf.clip_by_value(vpred-old_v, -self.clip_value, self.clip_value)
                    vf_losses1 = tf.square(vpred - r)
                    vf_losses2 = tf.square(vpredclipped - r)

                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    total_loss = surr_obj + self.vf_coef * vf_loss

                    gradients = inner_tape.gradient(total_loss, self.policy.trainable_variables)

                    if self.max_grad_norm is not None:
                        gradients, _grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)

                    self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

                    policy_loss += surr_obj.numpy()
                    value_loss += vf_loss.numpy()

        policy_losses.append(policy_loss / float(self.num_inner_grad_steps))
        value_losses.append(value_loss / float(self.num_inner_grad_steps))

        return policy_losses, value_losses


