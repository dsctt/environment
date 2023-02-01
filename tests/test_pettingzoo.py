from pdb import set_trace as T

from pettingzoo.test import parallel_api_test

import nmmo

def test_pettingzoo_api():
    config = nmmo.config.Default()
    config.PLAYERS = [nmmo.core.agent.Random]
    env = nmmo.Env(config)
    # TODO: disabled due to Env not implementing the correct PettinZoo step() API
    # parallel_api_test(env, num_cycles=1000)


if __name__ == '__main__':
    test_pettingzoo_api()