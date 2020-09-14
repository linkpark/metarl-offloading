import tensorflow as tf
import numpy as np
import itertools

# this is the tf graph version of reptile:
class MRLCO():
    def __init__(self,
                 policy,
                 meta_batch_size,
                 meta_sampler,
                 meta_sampler_process,
                 outer_lr=1e-4,
                 inner_lr=0.1,
                 num_inner_grad_steps=4,
                 clip_value = 0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.num_inner_grad_steps=num_inner_grad_steps
        self.policy = policy
        self.meta_sampler = meta_sampler
        self.meta_sampler_process = meta_sampler_process
        self.meta_batch_size = meta_batch_size
        self.update_numbers = 1

        #self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=self.lr, epsilon=1e-5)
        #self.inner_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.inner_lr)
        self.inner_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.inner_lr)
        self.outer_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.outer_lr)
        self.clip_value = clip_value
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # initialize the place hoder for each task place holder.
        self.new_logits = []
        self.decoder_inputs =[]
        self.old_logits = []
        self.actions = []
        self.obs = []
        self.vpred = []
        self.decoder_full_length = []

        self.old_v =[]
        self.advs = []
        self.r = []

        self.surr_obj = []
        self.vf_loss = []
        self.total_loss = []
        self._train = []

        self.build_graph()

    def build_graph(self):
        # build inner update for each tasks
        for i in range(self.meta_batch_size):
            self.new_logits.append(self.policy.meta_policies[i].network.decoder_logits)
            self.decoder_inputs.append(self.policy.meta_policies[i].decoder_inputs)
            self.old_logits.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, self.policy.action_dim], name='old_logits_ph_task_'+str(i)))
            self.actions.append(self.policy.meta_policies[i].decoder_targets)
            self.obs.append(self.policy.meta_policies[i].obs)
            self.vpred.append(self.policy.meta_policies[i].vf)
            self.decoder_full_length.append(self.policy.meta_policies[i].decoder_full_length)

            self.old_v.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='old_v_ph_task_'+str(i)))
            self.advs.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='advs_ph_task'+str(i)))
            self.r.append(tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='r_ph_task_'+str(i)))

            with tf.compat.v1.variable_scope("inner_update_parameters_task_"+str(i)) as scope:
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(self.actions[i], self.old_logits[i], self.new_logits[i])

                clipped_obj = tf.minimum(likelihood_ratio * self.advs[i] ,
                                         tf.clip_by_value(likelihood_ratio,
                                                          1.0 - self.clip_value,
                                                          1.0 + self.clip_value) * self.advs[i])
                self.surr_obj.append(-tf.reduce_mean(clipped_obj))

                vpredclipped = self.vpred[i] + tf.clip_by_value(self.vpred[i] - self.old_v[i], -self.clip_value, self.clip_value)
                vf_losses1 = tf.square(self.vpred[i] - self.r[i])
                vf_losses2 = tf.square(vpredclipped - self.r[i])

                self.vf_loss.append( .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) )

                self.total_loss.append( self.surr_obj[i] + self.vf_coef * self.vf_loss[i])

                params = self.policy.meta_policies[i].network.get_trainable_variables()

                grads_and_var = self.inner_optimizer.compute_gradients(self.total_loss[i], params)
                grads, var = zip(*grads_and_var)

                if self.max_grad_norm is not None:
                    # Clip the gradients (normalize)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_var = list(zip(grads, var))

                self._train.append(self.inner_optimizer.apply_gradients(grads_and_var))

        # Outer update for the parameters
        # feed in the parameters of inner policy network and update outer parameters.
        with tf.compat.v1.variable_scope("outer_update_parameters") as scope:
            core_network_parameters = self.policy.core_policy.get_trainable_variables()
            self.grads_placeholders = []

            for i, var in enumerate(core_network_parameters):
                self.grads_placeholders.append(tf.compat.v1.placeholder(shape=var.shape, dtype=var.dtype, name="grads_"+str(i)))

            outer_grads_and_var = list(zip(self.grads_placeholders, core_network_parameters))

            self._outer_train = self.outer_optimizer.apply_gradients(outer_grads_and_var)

    def UpdateMetaPolicy(self):
        # get the parameters value of the policy network
        sess = tf.compat.v1.get_default_session()

        for i in range(self.meta_batch_size):
            params_symbol = self.policy.meta_policies[i].get_trainable_variables()
            core_params_symble = self.policy.core_policy.get_trainable_variables()
            params = sess.run(params_symbol)
            core_params = sess.run(core_params_symble)

            update_feed_dict = {}

            # calcuate the gradient updates for the meta policy through first-order approximation.
            for i, core_var, meta_var in zip(itertools.count(), core_params, params):
                grads = (core_var - meta_var) / self.inner_lr / self.num_inner_grad_steps / self.meta_batch_size / self.update_numbers
                update_feed_dict[self.grads_placeholders[i]] = grads

            # update the meta policy parameters.
            _ = sess.run(self._outer_train, feed_dict=update_feed_dict)

        print("async core policy to meta-policy")
        self.policy.async_parameters()

    def UpdatePPOTarget(self, task_samples, batch_size=50):
        total_policy_losses = []
        total_value_losses = []
        for i in range(self.meta_batch_size):
            policy_losses, value_losses = self.UpdatePPOTargetPerTask(task_samples[i], i, batch_size)
            total_policy_losses.append(policy_losses)
            total_value_losses.append(value_losses)

        return total_policy_losses, total_value_losses

    def UpdatePPOTargetPerTask(self, task_samples, task_id, batch_size=50):
        policy_losses = []
        value_losses = []

        batch_number = int(task_samples['observations'].shape[0] / batch_size)
        self.update_numbers = batch_number
        #:q!
        # print("update number is: ", self.update_numbers)
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

        sess = tf.compat.v1.get_default_session()

        vf_loss = 0.0
        pg_loss = 0.0
        # copy_policy.set_weights(self.policy.get_weights())
        for i in range(self.num_inner_grad_steps):
            # action, old_logits, _ = copy_policy(observations)
            for old_logits, old_v, observations, actions, shift_actions, advs, r in zip(old_logits_batchs, oldvpred, observations_batchs, actions_batchs,
                                                                                        shift_action_batchs, advs_batchs, returns):
                decoder_full_length = np.array([observations.shape[1]] * observations.shape[0], dtype=np.int32)

                feed_dict = {self.old_logits[task_id]: old_logits, self.old_v[task_id]: old_v, self.obs[task_id]: observations, self.actions[task_id]: actions,
                            self.decoder_inputs[task_id]: shift_actions,
                             self.decoder_full_length[task_id]: decoder_full_length, self.advs[task_id]: advs, self.r[task_id]: r}

                _, value_loss, policy_loss = sess.run([self._train[task_id], self.vf_loss[task_id], self.surr_obj[task_id]], feed_dict=feed_dict)

                vf_loss += value_loss
                pg_loss += policy_loss

            vf_loss = vf_loss / float(self.num_inner_grad_steps)
            pg_loss = pg_loss / float(self.num_inner_grad_steps)

            value_losses.append(vf_loss)
            policy_losses.append(pg_loss)

        return policy_losses, value_losses
