import tensorflow as tf

class PPOReptile():
    """
    Provides ppo based reptile training
    """

    def __init__(self,
                 policy,
                 meta_sampler,
                 meta_sampler_process,
                 inner_lr=0.1,
                 outer_lr = 1e-3,
                 meta_batch_size=5,
                 num_inner_grad_steps=3,
                 clip_value = 0.2):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_batch_size = meta_batch_size
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process

        self.inner_optimizer = tf.keras.optimizers.SGD(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
        self.clip_value = clip_value

    def UpdateMetaPolicy(self):
        # update outer_parameters using reptile algorithm
        for i in range(self.meta_batch_size):
            outer_gradients = []

            for model_var, new_model_var in zip(self.policy.core_policy.trainable_variables,
                                                self.policy.meta_policies[i].trainable_variables):
                cur_grad = (tf.subtract(new_model_var, model_var)) / self.inner_lr / self.num_inner_grad_steps /self.meta_batch_size
                outer_gradients.append(cur_grad)

            self.outer_optimizer.apply_gradients(zip(outer_gradients, self.policy.core_policy.trainable_variables))

        self.policy.async_parameters()

    def UpdatePPOTarget(self, task_samples):
        losses = []
        for task_number in range(self.meta_batch_size):
            observations = task_samples[task_number]['observations']
            actions = tf.convert_to_tensor(task_samples[task_number]['actions'])
            advs = tf.convert_to_tensor(task_samples[task_number]['advantages'], dtype=tf.float32)

            action, old_logits, _ = self.policy.meta_policies[task_number](observations)

            # inner loss for ppo target:
            loss = 0.0
            for i in range(self.num_inner_grad_steps):
                with tf.GradientTape() as inner_tape:
                    _, new_logits, _ = self.policy.meta_policies[task_number](observations)
                    likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(actions, old_logits, new_logits)

                    clipped_obj = tf.minimum(likelihood_ratio * advs,
                                             tf.clip_by_value(likelihood_ratio,
                                                              1 - self.clip_value,
                                                              1 + self.clip_value) * advs)

                    surr_obj = -tf.reduce_mean(clipped_obj)
                    gradients = inner_tape.gradient(surr_obj, self.policy.meta_policies[task_number].trainable_variables)
                    self.inner_optimizer.apply_gradients(zip(gradients, self.policy.meta_policies[task_number].trainable_variables))

                    loss += surr_obj.numpy()

            losses.append(loss / float(self.num_inner_grad_steps))

        return losses