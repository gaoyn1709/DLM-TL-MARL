import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

'''
Sample Factory
2020年Intel和University of Sourhern Californi的论文《Sample Factory: Egocentric 3D Control From Pixels at 100000 FPS 
with Asynchronous Reinforcement Learning》提出了Sample Factory，它是为单机设计的高吞吐训练系统，基于
APPO(Asynchronous Proximal PolicyOptimization)算法。能在3D控制场景达到100000 FPS。

一个典型的强化学习中有三个主要的计算任务：环境仿真，模型推理和反向传播。设计的原则是让最慢的任务不要等其它的任务，因为系统的吞吐取决于
最慢任务的吞吐。每种任务对应一种类型的组件。组件之间通过高速的FIFO队列和共享内存通信。三种组件分别为：
1) Rollout worker：每个rollout worker可以有多个环境实例，与这些环境交互采集经验。Rollout worker一方面将经验通过共享内存给
policy worker，另一方面通过共享内存给learner。2）Policy worker：收集多个rollout worker来的状态，通过策略得到动作，通过共享内存传回给
rollout worker。3）Learner：前面两个一般有多个，只有这个是只有一份的。它通过共享内存从rollout worker拿到经验轨迹，更新策略，
然后通过GPU memory（因policy worker与learner都跑在GPU上）发往policy worker。Policy worker就可以用新的策略生成动作了。
Rollout worker和policy worker一起称为sampler子系统。为了解决rollout worker等待policy worker返回动作的问题，文中使用了
Double-Buffered Sampling的技巧。即在rollout worker上存一组环境 ，并分成两组，通过轮流来提高CPU利用率。
'''

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.global_state_shape
        self.obs_shape = args.observations_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    # 生成episodes数据，好几个rolloutWorker类同时产生数据
    def generate_episode(self, episode_num=None, evaluate=False):
        self.env.reset()
        terminated = False
        episode_reward = []  # cumulative rewards
        step = 0
        terminate = []
        obs = []
        acts = []
        padded = []
        total_episode_reward = 0
        total_fail_SD_num = 0

        self.agents.policy.init_hidden(1)   # 所有代理共享policy策略（一个evaluate网络，一个target网络）
        epsilon = 0 if evaluate else self.epsilon
        while not terminated and step < self.episode_limit:
            observations = self.env.get_obs()
            actions = []
            for agent_id in range(self.n_agents):
                avail_actions = self.env.get_avail_agent_actions(agent_id, step)
                # 一个agent的action是一个20长的list，而不是一个动作值
                action = self.agents.choose_action(observations[agent_id], agent_id, avail_actions, epsilon, evaluate)
                actions.append(action)
            reward, terminated, fail_SD_num = self.env.step(actions)
            total_fail_SD_num += fail_SD_num
            obs.append(observations)
            acts.append(np.reshape(actions, [self.n_agents, self.env.pre_SD_num]))
            episode_reward.append([reward])
            padded.append([0.])
            terminate.append([terminated])
            total_episode_reward += reward
            print(step)
            step += 1

        # last obs
        observations = self.env.get_obs()
        obs.append(observations)
        obs_next = obs[1:]
        obs = obs[:-1] # 去除最后一个

        # if step < self.episode_limit, padding
        for i in range(step, self.episode_limit):
            obs.append(np.zeros((self.n_agents, self.obs_shape)))
            obs_next.append(np.zeros((self.n_agents, self.obs_shape)))
            acts.append(np.zeros([self.n_agents, self.env.pre_SD_num]))
            episode_reward.append([0.0])
            padded.append([1.])
            terminate.append([1.])
        episode = dict( observations = obs.copy(),
                        observations_next = obs_next.copy(),
                        actions = acts.copy(),
                        rewards = episode_reward.copy(),
                        padded = padded.copy(),
                        terminate = terminate.copy()
                       )
        for key in episode:
            episode[key] = np.array([episode[key]])

        return episode, total_episode_reward/step, total_fail_SD_num, step