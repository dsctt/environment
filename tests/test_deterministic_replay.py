from pdb import set_trace as T
import unittest
from tqdm import tqdm

import pickle
import random

import nmmo

# try to reuse functions defined in test_determinism.py
import sys
sys.path.append('tests')

from test_determinism import TestEnv, TestConfig, serialize_actions, are_observations_equal


def load_replay_file(replay_file='tests/deterministic_replay_ver_1.6.0.7_seed_5554.pickle'):
   '''
      This function will try to load the passed pickle file.
      If the passed file is not found or not loaded properly, this function will generate a new file.
      It is possible to supply a different file, but it should also be supported by the test class (TODO)
   '''
   try:
      # load the pickle file
      with open(replay_file, 'rb') as handle:
         ref_data = pickle.load(handle)
      
         seed = ref_data['seed']
         config = ref_data['config']
         init_obs = ref_data['init_obs']
         actions = ref_data['actions']
         final_obs = ref_data['final_obs']
         final_npcs = ref_data['final_npcs']

         # test whether the loaded seed and config are valid
         print('[TestDetReplay] Testing whether the seed and config are valid')
         env_src = TestEnv(config, seed)
         obs = env_src.reset()
         assert are_observations_equal(obs, init_obs), "Something wrong with the provided pickle data"

   except:
      # generate the new data with a new env
      seed = random.randint(0, 10000)
      print('[TestDetReplay] Creating a new replay file with seed', seed)
      config = TestConfig()
      env_src = TestEnv(config, seed)
      init_obs = env_src.reset()

      test_horizon = 50
      actions = []
      print('Running', test_horizon, 'tikcs')
      for t in tqdm(range(test_horizon)):
         actions.append(serialize_actions(env_src.realm, env_src.actions))
         nxt_obs, _, _, _ = env_src.step({})
      final_obs = nxt_obs
      final_npcs = {}
      for nid, npc in list(env_src.realm.npcs.items()):
         final_npcs[nid] = npc.packet()
         del final_npcs[nid]['alive'] # to use the same 'are_observations_equal' function

      # save to the file
      with open(replay_file, 'wb') as handle:
         ref_data = {}
         ref_data['version'] = nmmo.__version__ # just in case
         ref_data['seed'] = seed
         ref_data['config'] = config
         ref_data['init_obs'] = init_obs
         ref_data['actions'] = actions
         ref_data['final_obs'] = final_obs
         ref_data['final_npcs'] = final_npcs

         pickle.dump(ref_data, handle)

   return seed, config, actions, final_obs, final_npcs


class TestDeterministicReplay(unittest.TestCase):
   @classmethod
   def setUpClass(cls):
      # TODO: allow providing the replay file by passing the file name to load_replay_file
      cls.seed, cls.config, cls.actions, cls.final_obs_src, cls.final_npcs_src = load_replay_file()
      cls.horizon = len(cls.actions)

      print('[TestDetReplay] Setting up the replication env with seed', cls.seed)
      env_rep = TestEnv(cls.config, seed=cls.seed)
      cls.init_obs_rep = env_rep.reset()
      print('Running', cls.horizon, 'tikcs')
      for t in tqdm(range(cls.horizon)):
         nxt_obs_rep, _, _, _ = env_rep.step(cls.actions[t])
      cls.final_obs_rep = nxt_obs_rep
      npcs_rep = {}
      for nid, npc in list(env_rep.realm.npcs.items()):
         npcs_rep[nid] = npc.packet()
         del npcs_rep[nid]['alive'] # to use the same 'are_observations_equal' function
      cls.final_npcs_rep = npcs_rep

   def test_compare_final_observations(self):
      self.assertTrue(are_observations_equal(self.final_obs_src, self.final_obs_rep))

   def test_compare_final_npcs(self):
      self.assertTrue(are_observations_equal(self.final_npcs_src, self.final_npcs_rep))


if __name__ == '__main__':
   unittest.main()

