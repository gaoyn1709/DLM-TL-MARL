import numpy as np
import os
from common.rollout import RolloutWorker
from common.replay_buffer import ReplayBuffer
from agent.agent import Agents
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        # no communication agent
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)

        self.args = args
        self.win_rates = [] # 记录200W次time_step里，每5000次评估一次模型的准确率
        self.episode_rewards = []   # episode进行一次模型验证，并保存最优模型
        self.total_fail_num = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps = 0
        train_steps = 0
        episodes = []
        evaluate_step = 0
        while time_steps < self.args.n_steps:   # 200W次
            for episode_id in range(self.args.n_episodes):
                episode, episode_reward, total_fail_SD_num, steps = self.rolloutWorker.generate_episode(episode_id)
                episodes.append(episode)
                time_steps += steps
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0) # 行不动列增加

            self.buffer.store_episode(episode_batch)

            for train_id in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                print('--------start to train-------')
                self.agents.train(mini_batch, train_steps)
                print('--------train end-------')
                train_steps += 1

            episode_reward, mean_fail_SD_num = self.evaluate()
            if evaluate_step == 0:
                print('============evaluation================')
            print('=========' + str(time_steps) + '=======' + str(episode_reward))
            self.episode_rewards.append(episode_reward)
            self.total_fail_num.append(mean_fail_SD_num)
            # self.plt(num)
            evaluate_step += 1

    def evaluate(self):
        episode_rewards = 0
        total_evaluate_fail_SD_num = 0
        for epoch in range(self.args.evaluate_epoch):   # 5000次
            episodes, episode_reward, total_fail_SD_num, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            total_evaluate_fail_SD_num += total_fail_SD_num
        return episode_rewards / self.args.evaluate_epoch, total_evaluate_fail_SD_num/self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 120])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.total_fail_num)), self.total_fail_num)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('mean_fail_num')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        plt.show()

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/mean_fail_num_{}'.format(num), self.total_fail_num)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









