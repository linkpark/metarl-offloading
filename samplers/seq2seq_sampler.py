from samplers.base import Sampler
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools
import tensorflow as tf

class Seq2SeqSampler(Sampler):
    """
    Sampler for PPO-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_meta_task,
                max_path_length,
                envs_per_task=None,
                parallel=False
                ):
        super(Seq2SeqSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.total_samples = rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0
        self.env = env

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = dict()

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.env.reset()

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            # obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            obs_per_task = np.array(obses)

            actions, logits, values = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            # actions = np.concatenate(actions)

            next_obses, rewards, dones, env_infos = self.env.step(actions)

            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            new_samples = 0
            for observation, action, logit, reward, value, finish_time in zip(obses, actions, logits,
                                                                       rewards, values, env_infos):
                running_paths["observations"] = observation
                running_paths["actions"] = action
                running_paths["logits"] = logit
                running_paths["rewards"] = reward
                running_paths["values"] = value
                running_paths["finish_time"] = finish_time
                # handling

                paths.append(dict(
                    observations=np.squeeze(np.asarray(running_paths["observations"])),
                    actions=np.squeeze(np.asarray(running_paths["actions"])),
                    logits=np.squeeze(np.asarray(running_paths["logits"])),
                    rewards=np.squeeze(np.asarray(running_paths["rewards"])),
                    values=np.squeeze(np.asarray(running_paths["values"])),
                    finish_time=np.squeeze(np.asarray(running_paths["finish_time"]))
                ))

                # if running path is done, add it to paths and empty the running path
                new_samples += len(running_paths["rewards"])
                running_paths = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths


def _get_empty_running_paths_dict():
    return dict()



if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.seq2seq_policy import Seq2SeqPolicy
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.linear_baseline import LinearFeatureBaseline
    from meta_algos.ppo_reptile import PPOReptile

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
    env.set_task(0)

    policy = Seq2SeqPolicy(name="pi",
                           obs_dim=17,
                           encoder_units=64,
                           decoder_units=64,
                           vocab_size=2)

    sampler = Seq2SeqSampler(env,
                             policy,
                             rollouts_per_meta_task=1,
                             max_path_length=5000,
                             envs_per_task=None,
                             parallel=False)

    baseline = LinearFeatureBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        paths = sampler.obtain_samples(log=True, log_prefix='Step')

        print(len(paths))
        print(paths[0]["values"])

        processing_data = sample_processor.process_samples(paths)

        print("processing data shape is:", processing_data['observations'].shape)
        #
        # print("processsing data action is: ", processing_data['actions'][0:10])
        # print("processing data reward is: ", processing_data['rewards'][0:10])
        # print("processing data action is: ", processing_data['actions'][10:20])
        # print("processing data reward is: ", processing_data['rewards'][10:20])





