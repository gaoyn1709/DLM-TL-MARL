import os

from Algorithm.IQL import IQL
from Algorithm.Quantum_network_routing import Quantum_network_routing
from runner import Runner
from common.arguments import get_common_args, get_mixer_args

if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        args = get_mixer_args(args)

        env = Quantum_network_routing()
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.global_state_shape = env_info["global_state_shape"]
        args.observations_shape = env_info["observations_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.n_SD = env_info["SD_num"]
        args.num_neighbor = env_info["num_neighbor"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break