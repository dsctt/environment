import unittest
import logging

# pylint: disable=import-error
from testhelpers import ScriptedAgentTestEnv, ScriptedAgentTestConfig

from scripted import baselines
from nmmo.io import action
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

TEST_HORIZON = 150
RANDOM_SEED = 985

LOGFILE = 'tests/action/test_ammo_use.log'

class TestAmmoUse(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # only use Combat agents
    cls.config = ScriptedAgentTestConfig()
    cls.config.PLAYERS = [baselines.Melee, baselines.Range, baselines.Mage]
    cls.config.PLAYER_N = 3
    cls.config.IMMORTAL = True
    
    # detailed logging for debugging
    cls.config.LOG_VERBOSE = False
    if cls.config.LOG_VERBOSE:
      logging.basicConfig(filename=LOGFILE, level=logging.INFO)

    # set up agents to test ammo use
    cls.policy = { 1:'Melee', 2:'Range', 3:'Mage' }
    cls.spawn_locs = { 1:(17, 17), 2:(17, 19), 3:(19, 19) }
    cls.ammo = { 1:Item.Scrap, 2:Item.Shaving, 3:Item.Shard }
    cls.ammo_quantity = 2

  def _change_spawn_pos(self, realm, ent_id, pos):
    # check if the position is valid
    assert realm.map.tiles[pos].habitable, "Given pos is not habitable."
    realm.players[ent_id].row.update(pos[0])
    realm.players[ent_id].col.update(pos[1])
    realm.players[ent_id].spawn_pos

  def _provide_item(self, realm, ent_id, item, level, quantity):
    realm.players[ent_id].inventory.receive(
      item(realm, level=level, quantity=quantity))
    
  def _setup_env(self):
    """ set up a new env and perform initial checks """
    env = ScriptedAgentTestEnv(self.config, seed=RANDOM_SEED)
    env.reset()
    for ent_id, pos in self.spawn_locs.items():
      self._change_spawn_pos(env.realm, ent_id, pos)
      self._provide_item(env.realm, ent_id, self.ammo[ent_id], 0, self.ammo_quantity)
    env.obs = env._compute_observations()

    # check if the agents are in specified positions
    for ent_id, pos in self.spawn_locs.items():
      self.assertEqual(env.realm.players[ent_id].policy, self.policy[ent_id])
      self.assertEqual(env.realm.players[ent_id].pos, pos)

      # agents see each other
      for other, pos in self.spawn_locs.items():
        self.assertTrue(other in env.obs[ent_id].entities.ids)

      # ammo instances are in the datastore and global item registry (realm)
      item_id = ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id
      self.assertTrue(ItemState.Query.by_id(env.realm.datastore, item_id) is not None)
      self.assertTrue(item_id in env.realm.items)

      # agents have ammo
      self.assertEqual(self.ammo[ent_id].ITEM_TYPE_ID,
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).type_id)
      self.assertEqual(self.ammo_quantity, # provided 2 ammos
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).quantity)

    return env

  def test_ammo_fire_all(self):
    env = self._setup_env()

    # First tick actions: USE (equip) ammo
    env.step({ ent_id: { action.Use: 
        { action.Item: ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id }
      } for ent_id in self.ammo })

    # check if the agents have equipped the ammo
    for ent_id in self.ammo:
      self.assertEqual(1, # True
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).equipped)

    # Second tick actions: ATTACK other agents using ammo
    #  NOTE that the agents are immortal
    env.step({ ent_id: { action.Attack: 
        { action.Style: env.realm.players[ent_id].agent.style[0],
          action.Target: (ent_id+1)%3+1 } }
        for ent_id in self.ammo })

    # check if the ammos were consumed
    ammo_ids = []
    for ent_id in self.ammo:
      item_info = ItemState.parse_array(env.obs[ent_id].inventory.values[0])
      self.assertEqual(1, item_info.quantity)
      ammo_ids.append(item_info.id)

    # Third tick actions: ATTACK again to use up all the ammo
    env.step({ ent_id: { action.Attack: 
        { action.Style: env.realm.players[ent_id].agent.style[0],
          action.Target: (ent_id+1)%3+1 } }
        for ent_id in self.ammo })

    # check if the ammos are depleted and the ammo slot is empty
    for ent_id in self.ammo:
      self.assertTrue(len(env.obs[ent_id].inventory.values) ==  0) # empty inventory
      self.assertTrue(env.realm.players[ent_id].inventory.equipment.ammunition.item == None)

    for item_id in ammo_ids:
      self.assertTrue(len(ItemState.Query.by_id(env.realm.datastore, item_id)) == 0)
      self.assertTrue(item_id not in env.realm.items)

    # DONE

  def test_cannot_use_listed_items(self):
    env = self._setup_env()

    sell_price = 1

    # First tick actions: SELL ammo
    env.step({ ent_id: { action.Sell: 
        { action.Item: ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id,
          action.Price: sell_price } }
        for ent_id in self.ammo })

    # check if the ammos were listed
    for ent_id in self.ammo:
      # ItemState data
      self.assertEqual(sell_price,
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).listed_price)
      # Exchange listing
      self.assertTrue(
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id \
        in env.realm.exchange._item_listings
      )

    # Second tick actions: USE ammo, which should NOT happen
    env.step({ ent_id: { action.Use: 
        { action.Item: ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id }
      } for ent_id in self.ammo })

    # check if the agents have equipped the ammo
    for ent_id in self.ammo:
      self.assertEqual(0, # False
        ItemState.parse_array(env.obs[ent_id].inventory.values[0]).equipped)

    # DONE

  def test_receive_extra_ammo_swap(self):
    env = self._setup_env()

    extra_ammo = 500

    for ent_id in self.policy:
      # provide extra scrap 
      self._provide_item(env.realm, ent_id, Item.Scrap, level=0, quantity=extra_ammo)
      self._provide_item(env.realm, ent_id, Item.Scrap, level=1, quantity=extra_ammo)

    # level up the agent 1 (Melee) to 2
    env.realm.players[1].skills.melee.level.update(2)
    env.obs = env._compute_observations()

    # check inventory
    for ent_id in self.ammo:
      inventory = { item.signature: item.quantity.val
                    for item in env.realm.players[ent_id].inventory.items }
      self.assertTrue( (Item.Scrap.ITEM_TYPE_ID, 0) in inventory )
      self.assertTrue( (Item.Scrap.ITEM_TYPE_ID, 1) in inventory )
      self.assertEqual( inventory[(Item.Scrap.ITEM_TYPE_ID, 1)], extra_ammo )
      if ent_id == 1:
        # if the ammo has the same signature, the quantity is added to the existing stack
        self.assertEqual( inventory[(Item.Scrap.ITEM_TYPE_ID, 0)], extra_ammo + self.ammo_quantity )
      else:
        # if the signature is different, it occupies a new inventory space
        self.assertEqual( inventory[(Item.Scrap.ITEM_TYPE_ID, 0)], extra_ammo )

    # First tick actions: USE (equip) ammo 0
    #   execute only the agent 1's action
    ent_id = 1
    env.step({ ent_id: { action.Use: 
        { action.Item: ItemState.parse_array(env.obs[ent_id].inventory.values[0]).id }}})

    # check if the agents have equipped the ammo 0
    self.assertTrue(ItemState.parse_array(env.obs[ent_id].inventory.values[0]).equipped == 1)
    self.assertTrue(ItemState.parse_array(env.obs[ent_id].inventory.values[1]).equipped == 0)

    # Second tick actions: USE (equip) ammo 1
    #   this should unequip 0 then equip 1
    env.step({ ent_id: { action.Use: 
        { action.Item: ItemState.parse_array(env.obs[ent_id].inventory.values[1]).id }}})

    # check if the agents have equipped the ammo 1
    self.assertTrue(ItemState.parse_array(env.obs[ent_id].inventory.values[0]).equipped == 0)
    self.assertTrue(ItemState.parse_array(env.obs[ent_id].inventory.values[1]).equipped == 1)

    # DONE


if __name__ == '__main__':
  unittest.main()
