import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.global_state_shape = args.global_state_shape
        self.observations_shape = args.observations_shape
        if args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        self.args = args
        self.num_SD = args.n_SD

    def choose_action(self, obs, agent_num, avail_actions, epsilon, evaluate):
        inputs = obs.copy()
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        avail_actions_ind = np.nonzero(avail_actions)[0]    # 求的avail_actions的非0位置
        # 返回的动作不应该是一个值
        action = []
        if np.random.uniform() < epsilon:
            # 从邻居里随机一个下一跳，但仍是所有SD对都有动作，无非是SD对组建路径的时候可能用不到这个节点
            for i in range(self.num_SD):
                action.append(avail_actions_ind[i])
            # action = np.random.choice(avail_actions_ind)
        else:
            # 当有资源限制的时候，必须对节点的SD对转发做出筛选，但暂且不考虑
            for i in range(self.num_SD):
                ii = i*self.args.num_neighbor
                action.append(int(torch.argmax(q_value[:, ii:ii+3])))
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)











