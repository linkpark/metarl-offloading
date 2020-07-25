import tensorflow as tf
import numpy as np
import time
from utils import logger


class Trainer():
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size

    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []

        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='Step_%d-' % itr)

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % itr)

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            #print("task losses: ", losses)
            print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % 100 == 0:
                self.policy.save_variables(save_path="./ppo_model/ppo_model_"+str(itr)+".ckpt")

        self.policy.save_variables(save_path="./ppo_model/ppo_model_final.ckpt")

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    # from policies.seq2seq_policy import Seq2SeqPolicy

    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                     "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                   ],
                                time_major=False)

    env.merge_graphs()
    env.set_task(0)

    action, finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(finish_time))
    print()
    finish_time, energy_cost = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time, energy_cost = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))

    policy = Seq2SeqPolicy(obs_dim=17,
                           encoder_units=128,
                           decoder_units=128,
                           vocab_size=2,
                           name="core_policy")

    sampler = Seq2SeqSampler(env,
                policy,
                rollouts_per_meta_task=1,
                max_path_length=760000,
                envs_per_task=None,
                parallel=False)

    baseline = ValueFunctionBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                           discount=0.99,
                                           gae_lambda=0.95,
                                           normalize_adv=True,
                                           positive_adv=False)

    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=5e-4,
               num_inner_grad_steps=3,
               clip_value=0.2,
               max_grad_norm=1.0)

    trainer = Trainer(algo = algo,
                    env=env,
                    sampler=sampler,

                    sample_processor=sample_processor,
                    policy=policy,
                    n_itr=1200,
                    start_itr=0,
                    batch_size=3800)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()

    import matplotlib.pyplot as plt

    x = np.arange(0, len(avg_ret), 1)

    plt.plot(x, avg_ret)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()

    x = np.arange(0, len(avg_pg_loss), 1)

    plt.plot(x, avg_pg_loss)
    plt.xlabel('episode')
    plt.ylabel('policy loss')
    plt.show()

    x = np.arange(0, len(avg_vf_loss), 1)

    plt.plot(x, avg_vf_loss)
    plt.xlabel('episode')
    plt.ylabel('value loss')
    plt.show()

    x = np.arange(0, len(avg_latencies), 1)

    plt.plot(x, avg_latencies)
    plt.xlabel('episode')
    plt.ylabel('avg_latency')
    plt.show()
