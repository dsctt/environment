from pdb import set_trace as T

import unittest
#import lovely_numpy
#lovely_numpy.monkey_patch()

import nmmo
from nmmo.entity.entity import Entity
from nmmo.core.realm import Realm
from nmmo.systems.item import Item


class TestApi(unittest.TestCase):
   env = nmmo.Env()
   config = env.config

   def test_observation_space(self):
      obs_space = self.env.observation_space(0)

      for entity in nmmo.Serialized.values():
         self.assertEqual(
            obs_space[entity.__name__]["Continuous"].shape[0], entity.N(self.config))

   def test_action_space(self):
      action_space = self.env.action_space(0)
      self.assertSetEqual(
         set(action_space.keys()), 
         set(nmmo.Action.edges(self.config)))

   def test_observations(self):
      obs = self.env.reset()
      
      self.assertEqual(obs.keys(), self.env.realm.players.entities.keys())
      
      for step in range(10):
         for player_id, player_obs in obs.items():
            self._validate_tiles(player_obs, self.env.realm)
            self._validate_entitites(player_obs, self.env.realm)
            self._validate_items(player_id, player_obs, self.env.realm)
         obs, _, _, _ = self.env.step({})

   def _validate_tiles(self, obs, realm: Realm):
      for tile_obs in obs["Tile"]["Continuous"]:
         tile = realm.map.tiles[int(tile_obs[2]), int(tile_obs[3])]
         self.assertListEqual(list(tile_obs),
            [tile.nEnts.val, tile.index.val, tile.r.val, tile.c.val])

   def _validate_entitites(self, obs, realm: Realm):
      for entity_obs in obs["Entity"]["Continuous"]:
         if entity_obs[0] == 0: continue
         entity: Entity = realm.entity(entity_obs[1])
         self.assertListEqual(list(entity_obs), [
            1, 
            entity.entID, 
            entity.attackerID.val,
            entity.base.level.val, 
            entity.base.item_level.val, 
            entity.base.comm.val,
            entity.base.population.val,
            entity.base.r.val, 
            entity.base.c.val,
            entity.history.damage.val,
            entity.history.timeAlive.val,
            entity.status.freeze.val,
            entity.base.gold.val,
            entity.resources.health.val,
            entity.resources.food.val,
            entity.resources.water.val,
            entity.skills.melee.level.val,
            entity.skills.range.level.val,
            entity.skills.mage.level.val,
            (entity.skills.fishing.level.val if entity.isPlayer else 0),
            (entity.skills.herbalism.level.val if entity.isPlayer else 0),
            (entity.skills.prospecting.level.val if entity.isPlayer else 0),
            (entity.skills.carving.level.val if entity.isPlayer else 0),
            (entity.skills.alchemy.level.val if entity.isPlayer else 0),
         ], f"Mismatch for Entity {entity.entID}")

   def _validate_items(self, player_id, obs, realm: Realm):
      item_refs = realm.players[player_id].inventory._item_references
      item_obs = obs["Item"]["Continuous"]
      # Something like this?
      #assert len(item_refs) == len(item_obs)
      for ob, item in zip(item_obs, item_refs):
         self.assertListEqual(list(ob), [
            item.instanceID,
            item.index.val,
            item.level.val,
            item.capacity.val,
            item.quantity.val,
            item.tradable.val,
            item.melee_attack.val,
            item.range_attack.val,
            item.mage_attack.val,
            item.melee_defense.val,
            item.range_defense.val,
            item.mage_defense.val,
            item.health_restore.val,
            item.resource_restore.val,
            item.price.val,
            item.equipped.val
         ], f"Mismatch for Item {item.instanceID}")

if __name__ == '__main__':
    unittest.main()
