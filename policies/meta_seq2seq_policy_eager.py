import tensorflow as tf
import numpy as np

from policies.networks.seq2seq import Seq2SeqNetwork
from policies.distributions.categorical_pd import CategoricalPd

class Seq2SeqPolicy(tf.keras.Model):
    def __init__(self, obs_dim, encoder_units, decoder_units, vocab_size, value_network_dimension=2):
        super(Seq2SeqPolicy, self).__init__()
        self.obs_dim = obs_dim
        # stochastic action dim
        self.action_dim = vocab_size
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

        self.seq2seq_network = Seq2SeqNetwork(encoder_units, decoder_units, vocab_size)
        # self.q_net = tf.keras.layers.Dense(value_network_dimension, activation=tf.keras.activations.relu)
        self.q_flatten_layer = tf.keras.layers.Dense(self.action_dim)

        self._dist = CategoricalPd(vocab_size)

        self.initialize_variables()


    def call(self, enc_input, decoder_input):
        logits = self.seq2seq_network(enc_input, decoder_input)

        q_value = self.q_flatten_layer(logits)
        pi = tf.nn.softmax(logits)
        v_value = tf.reduce_sum(pi * q_value, axis=-1)

        return logits, v_value

    def get_actions(self, x):
        actions, logits =  self.seq2seq_network.sample(x)

        q_value = self.q_flatten_layer(logits)
        pi = tf.nn.softmax(logits)
        v_value = tf.reduce_sum(pi * q_value, axis=-1)

        return actions.numpy(), logits.numpy(), v_value.numpy()

    def greedy_select(self, x):
        action, decoder_output_logits = self.seq2seq_network.greedy_select(x)

        return action.numpy()

    def initialize_variables(self):
        x = np.ones((1, 1, self.obs_dim), dtype=np.float32)
        _, _, _ = self.get_actions(x)

    @property
    def distribution(self):
        return self._dist


class MetaSeq2SeqPolicy():
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units,
                 vocab_size, value_network_dimension):
        self.meta_batch_size = meta_batch_size
        self.core_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                 vocab_size, value_network_dimension)

        self.meta_policies = []
        for i in range(meta_batch_size):
            meta_policy = Seq2SeqPolicy(obs_dim, encoder_units, decoder_units,
                 vocab_size, value_network_dimension)

            self.meta_policies.append(meta_policy)

        self._dist = CategoricalPd(vocab_size)

        self.async_parameters()

    def get_actions(self, observations):
        assert len(observations) == self.meta_batch_size

        meta_actions = []
        meta_logits = []
        meta_v_values = []
        for i, obser_per_task in enumerate(observations):
            action, logits, v_value = self.meta_policies[i](obser_per_task)

            meta_actions.append(action.numpy())
            meta_logits.append(logits.numpy())
            meta_v_values.append(v_value.numpy())

        return meta_actions, meta_logits, meta_v_values

    def async_parameters(self):
        var_weights = self.core_policy.get_weights()
        for i in range(self.meta_batch_size):
            self.meta_policies[i].set_weights(var_weights)

    @property
    def distribution(self):
        return self._dist

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.linear_baseline import LinearFeatureBaseline

    tf.compat.v1.enable_eager_execution()


    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=10,
                                graph_number=10,
                                graph_file_paths=[
                                    "D:\\Code\\MetaRL-offloading-Reptile\\env\\mec_offloaing_envs\\data\\offload_random10\\random.10.",
                                    "D:\\Code\\MetaRL-offloading-Reptile\\env\\mec_offloaing_envs\\data\\offload_random10\\random.10.",
                                    "D:\\Code\\MetaRL-offloading-Reptile\\env\\mec_offloaing_envs\\data\\offload_random10\\random.10.",
                                    "D:\\Code\\MetaRL-offloading-Reptile\\env\\mec_offloaing_envs\\data\\offload_random10\\random.10.",
                                    "D:\\Code\\MetaRL-offloading-Reptile\\env\\mec_offloaing_envs\\data\\offload_random10\\random.10."],
                                time_major=False)

    baseline = LinearFeatureBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=5, obs_dim=17, encoder_units=1, decoder_units=2,
                               vocab_size=2, value_network_dimension=2)

    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=5,
        max_path_length=1000,
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=False)


    paths_meta = sampler.obtain_samples(log=True, log_prefix='1')
    samples_data = sample_processor.process_samples(paths_meta, log=False, log_prefix='step')

    print(samples_data[0]['rewards'].shape)
    print(samples_data[0]['finish_time'].shape)
    print(samples_data[0]['finish_time'][0])

    ret = np.array([])
    for i in range(5):
        ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)

    print(ret.shape)




