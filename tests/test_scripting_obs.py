from pdb import set_trace as T
import unittest
from tqdm import tqdm

import nmmo
from scripted import baselines

TEST_HORIZON = 5


class Config(nmmo.config.Small, nmmo.config.AllGameSystems):

    RENDER = False
    SPECIALIZE = True
    PLAYERS = [
            baselines.Fisher, baselines.Herbalist, baselines.Prospector, baselines.Carver, baselines.Alchemist,
            baselines.Melee, baselines.Range, baselines.Mage]


class TestScriptingObservation(unittest.TestCase):
   @classmethod
   def setUpClass(cls):
      cls.config = Config()
      cls.env = nmmo.Env(cls.config)
      cls.env.reset()

      print('Running', TEST_HORIZON, 'tikcs')
      for t in tqdm(range(TEST_HORIZON)):
         cls.env.step({})

      cls.obs, _ = cls.env.realm.dataframe.get(cls.env.realm.players)

   def test_observation_agent(self):
      for playerID in self.obs.keys():
         ob = nmmo.scripting.Observation(self.config, self.obs[playerID])
         agent = ob.agent

         # player's entID must match
         self.assertEqual(playerID, nmmo.scripting.Observation.attribute(agent, nmmo.Serialized.Entity.ID))

   def test_observation_tile(self):
      vision = self.config.PLAYER_VISION_RADIUS

      for playerID in self.obs.keys():
         ob = nmmo.scripting.Observation(self.config, self.obs[playerID])
         agent = ob.agent

         # the current player's location
         r_cent = nmmo.scripting.Observation.attribute(agent, nmmo.Serialized.Entity.R)
         c_cent = nmmo.scripting.Observation.attribute(agent, nmmo.Serialized.Entity.C)

         for r_delta in range(-vision, vision+1):
            for c_delta in range(-vision, vision+1):
               tile = ob.tile(r_delta, c_delta)

               # tile's coordinate must match
               self.assertEqual(r_cent + r_delta, nmmo.scripting.Observation.attribute(tile, nmmo.Serialized.Tile.R))
               self.assertEqual(c_cent + c_delta, nmmo.scripting.Observation.attribute(tile, nmmo.Serialized.Tile.C))

if __name__ == '__main__':
   unittest.main()

