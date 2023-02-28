import unittest
import logging

# pylint: disable=import-error
from testhelpers import ScriptedAgentTestEnv, ScriptedAgentTestConfig

from scripted import baselines
from nmmo.io import action
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState
from nmmo.entity.entity import EntityState

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
    # 1 cannot hit 3, 2 can hit 1, 3 cannot hit 2
    cls.spawn_locs = { 1:(17, 17), 2:(17, 19), 3:(21, 21) }
    cls.ammo = { 1:Item.Scrap, 2:Item.Shaving, 3:Item.Shard }
    cls.ammo_quantity = 2

    # items to provide
    cls.item_sig = {}
    for ent_id in cls.policy:
      cls.item_sig[ent_id] = []
      for item in [cls.ammo[ent_id], Item.Top, Item.Gloves, Item.Ration, Item.Poultice]:
        for lvl in [0, 3]:
          cls.item_sig[ent_id].append((item, lvl))

  def _change_spawn_pos(self, realm, ent_id, pos):
    # check if the position is valid
    assert realm.map.tiles[pos].habitable, "Given pos is not habitable."
    realm.players[ent_id].row.update(pos[0])
    realm.players[ent_id].col.update(pos[1])
    realm.players[ent_id].spawn_pos = pos

  def _provide_item(self, realm, ent_id, item, level, quantity):
    realm.players[ent_id].inventory.receive(
      item(realm, level=level, quantity=quantity))

  def _setup_env(self):
    """ set up a new env and perform initial checks """
    env = ScriptedAgentTestEnv(self.config, seed=RANDOM_SEED)
    env.reset()

    for ent_id, pos in self.spawn_locs.items():
      self._change_spawn_pos(env.realm, ent_id, pos)
      env.realm.players[ent_id].gold.update(5)
      for item_sig in self.item_sig[ent_id]:
        if item_sig[0] == self.ammo[ent_id]:
          self._provide_item(env.realm, ent_id, item_sig[0], item_sig[1], self.ammo_quantity)
        else:
          self._provide_item(env.realm, ent_id, item_sig[0], item_sig[1], 1)
    env.obs = env._compute_observations()

    # check if the agents are in specified positions
    for ent_id, pos in self.spawn_locs.items():
      self.assertEqual(env.realm.players[ent_id].policy, self.policy[ent_id])
      self.assertEqual(env.realm.players[ent_id].pos, pos)

      # agents see each other
      for other, pos in self.spawn_locs.items():
        self.assertTrue(other in env.obs[ent_id].entities.ids)

      # ammo instances are in the datastore and global item registry (realm)
      inventory = env.obs[ent_id].inventory
      self.assertTrue(inventory.len == len(self.item_sig[ent_id]))
      for inv_idx in range(inventory.len):
        item_id = inventory.id(inv_idx)
        self.assertTrue(ItemState.Query.by_id(env.realm.datastore, item_id) is not None)
        self.assertTrue(item_id in env.realm.items)

      # agents have ammo
      for lvl in [0, 3]:
        inv_idx = inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, lvl)
        self.assertTrue(inv_idx is not None)
        self.assertEqual(self.ammo_quantity, # provided 2 ammos
          ItemState.parse_array(inventory.values[inv_idx]).quantity)

      # check ActionTargets
      gym_obs = env.obs[ent_id].to_gym()
      
      # ATTACK Target mask
      entities = env.obs[ent_id].entities.ids
      mask = gym_obs['ActionTargets'][action.Attack][action.Target][:len(entities)] > 0
      if ent_id == 1: 
        self.assertTrue(2 in entities[mask])
        self.assertTrue(3 not in entities[mask])
      if ent_id == 2:
        self.assertTrue(1 in entities[mask])
        self.assertTrue(3 not in entities[mask])
      if ent_id == 3:
        self.assertTrue(1 not in entities[mask])
        self.assertTrue(2 not in entities[mask])

      # USE InventoryItem mask
      inventory = env.obs[ent_id].inventory
      mask = gym_obs['ActionTargets'][action.Use][action.InventoryItem][:inventory.len] > 0
      for item_sig in self.item_sig[ent_id]:
        inv_idx = inventory.sig(item_sig[0].ITEM_TYPE_ID, item_sig[1])
        if item_sig[1] == 0:
          # items that can be used 
          self.assertTrue(inventory.id(inv_idx) in inventory.ids[mask])
        else:
          # items that are too high to use
          self.assertTrue(inventory.id(inv_idx) not in inventory.ids[mask])

      # SELL InventoryItem mask
      mask = gym_obs['ActionTargets'][action.Sell][action.InventoryItem][:inventory.len] > 0
      for item_sig in self.item_sig[ent_id]:
        inv_idx = inventory.sig(item_sig[0].ITEM_TYPE_ID, item_sig[1])
        # the agent can sell anything now
        self.assertTrue(inventory.id(inv_idx) in inventory.ids[mask])

      # BUY MarketItem mask -- there is nothing on the market, so mask should be all 0
      market = env.obs[ent_id].market
      mask = gym_obs['ActionTargets'][action.Buy][action.MarketItem][:market.len] > 0
      self.assertTrue(len(market.ids[mask]) == 0)

    return env

  def test_ammo_fire_all(self):
    env = self._setup_env()

    # First tick actions: USE (equip) level-0 ammo
    env.step({ ent_id: { action.Use: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0) }
      } for ent_id in self.ammo })

    # check if the agents have equipped the ammo
    for ent_id in self.ammo:
      gym_obs = env.obs[ent_id].to_gym()
      inventory = env.obs[ent_id].inventory
      inv_idx = inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0)
      self.assertEqual(1, # True
        ItemState.parse_array(inventory.values[inv_idx]).equipped)

      # check SELL InventoryItem mask -- one cannot sell equipped item
      mask = gym_obs['ActionTargets'][action.Sell][action.InventoryItem][:inventory.len] > 0
      self.assertTrue(inventory.id(inv_idx) not in inventory.ids[mask])

    # Second tick actions: ATTACK other agents using ammo
    #  NOTE that the agents are immortal
    #  NOTE that agents 1 & 3's attack are invalid due to out-of-range
    env.step({ ent_id: { action.Attack: 
        { action.Style: env.realm.players[ent_id].agent.style[0],
          action.Target: (ent_id+1)%3+1 } }
        for ent_id in self.ammo })

    # check if the ammos were consumed
    ammo_ids = []
    for ent_id in self.ammo:
      inventory = env.obs[ent_id].inventory
      inv_idx = inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0)
      item_info = ItemState.parse_array(inventory.values[inv_idx])
      if ent_id == 2:
        # only agent 2's attack is valid and consume ammo
        self.assertEqual(self.ammo_quantity - 1, item_info.quantity)
        ammo_ids.append(inventory.id(inv_idx))
      else:
        self.assertEqual(self.ammo_quantity, item_info.quantity)

    # Third tick actions: ATTACK again to use up all the ammo, except agent 3
    #  NOTE that agent 3's attack command is invalid due to out-of-range
    env.step({ ent_id: { action.Attack: 
        { action.Style: env.realm.players[ent_id].agent.style[0],
          action.Target: (ent_id+1)%3+1 } }
        for ent_id in self.ammo })

    # check if the ammos are depleted and the ammo slot is empty
    ent_id = 2
    self.assertTrue(env.obs[ent_id].inventory.len == len(self.item_sig[ent_id]) - 1)
    self.assertTrue(env.realm.players[ent_id].inventory.equipment.ammunition.item == None)

    for item_id in ammo_ids:
      self.assertTrue(len(ItemState.Query.by_id(env.realm.datastore, item_id)) == 0)
      self.assertTrue(item_id not in env.realm.items)

    # invalid attacks
    for ent_id in [1, 3]:
      # agent 3 gathered shaving, so the item count increased
      #self.assertTrue(env.obs[ent_id].inventory.len == len(self.item_sig[ent_id]))
      self.assertTrue(env.realm.players[ent_id].inventory.equipment.ammunition.item is not None)

    # DONE

  def test_cannot_use_listed_items(self):
    env = self._setup_env()

    sell_price = 1

    # provide extra scrap to range to make its inventory full
    # but level-0 scrap overlaps with the listed item
    ent_id = 2
    self._provide_item(env.realm, ent_id, Item.Scrap, level=0, quantity=3)
    self._provide_item(env.realm, ent_id, Item.Scrap, level=1, quantity=3)

    # provide extra scrap to mage to make its inventory full
    # there will be no overlapping item
    ent_id = 3
    self._provide_item(env.realm, ent_id, Item.Scrap, level=5, quantity=3)
    self._provide_item(env.realm, ent_id, Item.Scrap, level=7, quantity=3)
    env.obs = env._compute_observations()

    # First tick actions: SELL level-0 ammo
    env.step({ ent_id: { action.Sell: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0),
          action.Price: sell_price } }
        for ent_id in self.ammo })

    # check if the ammos were listed
    for ent_id in self.ammo:
      gym_obs = env.obs[ent_id].to_gym()
      inventory = env.obs[ent_id].inventory
      inv_idx = inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0)
      item_info = ItemState.parse_array(inventory.values[inv_idx])
      # ItemState data
      self.assertEqual(sell_price, item_info.listed_price)
      # Exchange listing
      self.assertTrue(item_info.id in env.realm.exchange._item_listings)
      self.assertTrue(item_info.id in env.obs[ent_id].market.ids)

      # check SELL InventoryItem mask -- one cannot sell listed item
      mask = gym_obs['ActionTargets'][action.Sell][action.InventoryItem][:inventory.len] > 0
      self.assertTrue(inventory.id(inv_idx) not in inventory.ids[mask])

      # check USE InventoryItem mask -- one cannot use listed item
      mask = gym_obs['ActionTargets'][action.Use][action.InventoryItem][:inventory.len] > 0
      self.assertTrue(inventory.id(inv_idx) not in inventory.ids[mask])

      # check BUY MarketItem mask -- there should be two ammo items in the market
      mask = gym_obs['ActionTargets'][action.Buy][action.MarketItem][:inventory.len] > 0
      # agent 1 has inventory space
      if ent_id == 1: self.assertTrue(sum(mask) == 2)
      # agent 2's inventory is full but can buy level-0 scrap (existing ammo)
      if ent_id == 2: self.assertTrue(sum(mask) == 1)
      # agent 3's inventory is full without overlapping ammo
      if ent_id == 3: self.assertTrue(sum(mask) == 0)

    # Second tick actions: USE ammo, which should NOT happen
    env.step({ ent_id: { action.Use: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0) }
      } for ent_id in self.ammo })

    # check if the agents have equipped the ammo
    for ent_id in self.ammo:
      inventory = env.obs[ent_id].inventory
      inv_idx = inventory.sig(self.ammo[ent_id].ITEM_TYPE_ID, 0)
      self.assertEqual(0, # False
        ItemState.parse_array(inventory.values[inv_idx]).equipped)

    # DONE

  def test_receive_extra_ammo_swap(self):
    env = self._setup_env()

    extra_ammo = 500
    scrap_lvl0 = (Item.Scrap.ITEM_TYPE_ID, 0)
    scrap_lvl1 = (Item.Scrap.ITEM_TYPE_ID, 1)
    scrap_lvl3 = (Item.Scrap.ITEM_TYPE_ID, 3)

    for ent_id in self.policy:
      # provide extra scrap 
      self._provide_item(env.realm, ent_id, Item.Scrap, level=0, quantity=extra_ammo)
      self._provide_item(env.realm, ent_id, Item.Scrap, level=1, quantity=extra_ammo)

    # level up the agent 1 (Melee) to 2
    env.realm.players[1].skills.melee.level.update(2)
    env.obs = env._compute_observations()

    # check inventory
    for ent_id in self.ammo:
      # realm data
      inv_realm = { item.signature: item.quantity.val
                    for item in env.realm.players[ent_id].inventory.items
                    if isinstance(item, Item.Stack) }
      self.assertTrue( scrap_lvl0 in inv_realm )
      self.assertTrue( scrap_lvl1 in inv_realm )
      self.assertEqual( inv_realm[scrap_lvl1], extra_ammo )

      # item datastore
      inv_obs = env.obs[ent_id].inventory
      self.assertTrue(inv_obs.sig(*scrap_lvl0) is not None)
      self.assertTrue(inv_obs.sig(*scrap_lvl1) is not None)
      self.assertEqual( extra_ammo,
        ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl1)]).quantity)
      if ent_id == 1:
        # if the ammo has the same signature, the quantity is added to the existing stack
        self.assertEqual( inv_realm[scrap_lvl0], extra_ammo + self.ammo_quantity )
        self.assertEqual( extra_ammo + self.ammo_quantity,
          ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl0)]).quantity)
        # so there should be 1 more space
        self.assertEqual( inv_obs.len, self.config.ITEM_INVENTORY_CAPACITY - 1)

      else:
        # if the signature is different, it occupies a new inventory space
        self.assertEqual( inv_realm[scrap_lvl0], extra_ammo )
        self.assertEqual( extra_ammo,
          ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl0)]).quantity)
        # thus the inventory is full
        self.assertEqual( inv_obs.len, self.config.ITEM_INVENTORY_CAPACITY)

      if ent_id == 1:
        gym_obs = env.obs[ent_id].to_gym()
        # check USE InventoryItem mask
        mask = gym_obs['ActionTargets'][action.Use][action.InventoryItem][:inv_obs.len] > 0
        # level-2 melee should be able to use level-0, level-1 scrap but not level-3
        self.assertTrue(inv_obs.id(inv_obs.sig(*scrap_lvl0)) in inv_obs.ids[mask])
        self.assertTrue(inv_obs.id(inv_obs.sig(*scrap_lvl1)) in inv_obs.ids[mask])
        self.assertTrue(inv_obs.id(inv_obs.sig(*scrap_lvl3)) not in inv_obs.ids[mask])

    # First tick actions: USE (equip) level-0 ammo
    #   execute only the agent 1's action
    ent_id = 1
    env.step({ ent_id: { action.Use: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(*scrap_lvl0) } }})

    # check if the agents have equipped the ammo 0
    inv_obs = env.obs[ent_id].inventory
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl0)]).equipped == 1)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl1)]).equipped == 0)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl3)]).equipped == 0)

    # Second tick actions: USE (equip) level-1 ammo
    #   this should unequip level-0 then equip level-1 ammo
    env.step({ ent_id: { action.Use: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(*scrap_lvl1) } }})

    # check if the agents have equipped the ammo 1
    inv_obs = env.obs[ent_id].inventory
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl0)]).equipped == 0)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl1)]).equipped == 1)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl3)]).equipped == 0)

    # Third tick actions: USE (equip) level-3 ammo
    #   this should ignore USE action and leave level-1 ammo equipped
    env.step({ ent_id: { action.Use: 
        { action.InventoryItem: env.obs[ent_id].inventory.sig(*scrap_lvl3) } }})

    # check if the agents have equipped the ammo 1
    inv_obs = env.obs[ent_id].inventory
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl0)]).equipped == 0)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl1)]).equipped == 1)
    self.assertTrue(ItemState.parse_array(inv_obs.values[inv_obs.sig(*scrap_lvl3)]).equipped == 0)

    # DONE


if __name__ == '__main__':
  unittest.main()
