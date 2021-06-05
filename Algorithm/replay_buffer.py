import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.global_state_shape = self.args.global_state_shape
        self.observations_shape = self.args.observations_shape
        self.size = self.args.buffer_size   # 5e3 5000大小
        self.episode_limit = self.args.episode_limit    # quantum network环境传过来的，1000次
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info, 存取的是所有代理的信息
        self.buffers = {'observations': np.empty([self.size, self.episode_limit, self.n_agents, self.observations_shape]),
                        'observations_next': np.empty([self.size, self.episode_limit, self.n_agents, self.observations_shape]),
                        'actions': np.empty([self.size, self.episode_limit, self.n_agents, (int)(self.n_actions/self.args.num_neighbor)]),
                        'rewards': np.empty([self.size, self.episode_limit, 1]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['observations'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            # store the informations
            self.buffers['observations'][idxs] = episode_batch['observations']
            self.buffers['observations_next'][idxs] = episode_batch['observations_next']
            self.buffers['actions'][idxs] = episode_batch['actions']
            self.buffers['rewards'][idxs] = episode_batch['rewards']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminate']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
