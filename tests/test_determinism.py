from pdb import set_trace as T
import unittest
from tqdm import tqdm
import numpy as np
import random

import nmmo

from scripted import baselines

from testhelpers import TestEnv, TestConfig, serialize_actions, are_observations_equal

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 10000)


class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.horizon = TEST_HORIZON
        cls.rand_seed = RANDOM_SEED
        cls.config = TestConfig()

        print('[TestDeterminism] Setting up the reference env with seed', cls.rand_seed)
        env_src = TestEnv(cls.config, seed=cls.rand_seed)
        actions_src = []
        cls.init_obs_src = env_src.reset()
        print('Running', cls.horizon, 'tikcs')
        for t in tqdm(range(cls.horizon)):
            actions_src.append(serialize_actions(env_src, env_src.actions))
            nxt_obs_src, _, _, _ = env_src.step({})
        cls.final_obs_src = nxt_obs_src
        cls.actions_src = actions_src
        npcs_src = {}
        for nid, npc in list(env_src.realm.npcs.items()):
            npcs_src[nid] = npc.packet()
            del npcs_src[nid]['alive'] # to use the same 'are_observations_equal' function
        cls.final_npcs_src = npcs_src

        print('[TestDeterminism] Setting up the replication env with seed', cls.rand_seed)
        env_rep = TestEnv(cls.config, seed=cls.rand_seed)
        actions_rep = []
        cls.init_obs_rep = env_rep.reset()
        print('Running', cls.horizon, 'tikcs')
        for t in tqdm(range(cls.horizon)):
            actions_rep.append(serialize_actions(env_rep, env_rep.actions))
            nxt_obs_rep, _, _, _ = env_rep.step({})
        cls.final_obs_rep = nxt_obs_rep
        cls.actions_rep = actions_rep
        npcs_rep = {}
        for nid, npc in list(env_rep.realm.npcs.items()):
            npcs_rep[nid] = npc.packet()
            del npcs_rep[nid]['alive'] # to use the same 'are_observations_equal' function
        cls.final_npcs_rep = npcs_rep
        
    def test_func_are_observations_equal(self):
        # are_observations_equal CANNOT be replaced with assertDictEqual
        self.assertTrue(are_observations_equal(self.init_obs_src, self.init_obs_src))
        self.assertTrue(are_observations_equal(self.final_obs_src, self.final_obs_src))
        #self.assertDictEqual(self.final_obs_src, self.final_obs_src)

    def test_func_are_actions_equal(self):
        # are_actions_equal can be replaced with assertDictEqual
        for t in range(len(self.actions_src)):
            #self.assertTrue(are_actions_equal(self.actions_src[t], self.actions_src[t]))
            self.assertDictEqual(self.actions_src[t], self.actions_src[t])

    def test_compare_initial_observations(self):
        self.assertTrue(are_observations_equal(self.init_obs_src, self.init_obs_rep))

    def test_compare_actions(self):
        for t in range(len(self.actions_src)):
            self.assertDictEqual(self.actions_src[t], self.actions_rep[t])

    def test_compare_final_observations(self):
        self.assertTrue(are_observations_equal(self.final_obs_src, self.final_obs_rep))

    def test_compare_final_npcs(self)        :
        self.assertTrue(are_observations_equal(self.final_npcs_src, self.final_npcs_rep))


if __name__ == '__main__':
    unittest.main()