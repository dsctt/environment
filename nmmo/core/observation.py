from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
import nmmo.systems.item as item_system
from nmmo.io import action
from nmmo.lib import material

class Observation:
  def __init__(self,
    config,
    agent_id: int,
    tiles,
    entities,
    inventory,
    market) -> None:

    self.config = config
    self.agent_id = agent_id

    self.tiles = tiles[0:config.MAP_N_OBS]

    entities = entities[0:config.PLAYER_N_OBS]
    entity_ids = entities[:,EntityState.State.attr_name_to_col["id"]]
    entity_pos = entities[:,[EntityState.State.attr_name_to_col["row"],
                             EntityState.State.attr_name_to_col["col"]]]
    self.entities = SimpleNamespace(
        values = entities,
        ids = entity_ids,
        len = len(entity_ids),
        id = lambda i: entity_ids[i] if i < len(entity_ids) else None,
        index = lambda val: np.nonzero(entity_ids == val)[0][0] if val in entity_ids else None,
        pos = entity_pos,
        # for the distance function, see io/action.py, Attack.call(), line 222
        dist = lambda pos: np.max(np.abs(entity_pos - np.array(pos)), axis=1),
    )

    if config.ITEM_SYSTEM_ENABLED:
      inventory = inventory[0:config.INVENTORY_N_OBS]
      inv_ids = inventory[:,ItemState.State.attr_name_to_col["id"]]
      inv_type = inventory[:,ItemState.State.attr_name_to_col["type_id"]]
      inv_level = inventory[:,ItemState.State.attr_name_to_col["level"]]
      self.inventory = SimpleNamespace(
        values = inventory,
        ids = inv_ids,
        len = len(inv_ids),
        id = lambda i: inv_ids[i] if i < len(inv_ids) else None,
        index = lambda val: np.nonzero(inv_ids == val)[0][0] if val in inv_ids else None,
        sig = lambda itm_type, level:
                np.nonzero((inv_type == itm_type) & (inv_level == level))[0][0]
                if (itm_type in inv_type) and (level in inv_level) else None
      )
    else:
      assert inventory.size == 0

    if config.EXCHANGE_SYSTEM_ENABLED:
      market = market[0:config.MARKET_N_OBS]
      market_ids = market[:,ItemState.State.attr_name_to_col["id"]]
      self.market = SimpleNamespace(
        values = market,
        ids = market_ids,
        len = len(market_ids),
        id = lambda i: market_ids[i] if i < len(market_ids) else None,
        index = lambda val: np.nonzero(market_ids == val)[0][0] if val in market_ids else None
      )
    else:
      assert market.size == 0

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def tile(self, r_delta, c_delta):
    '''Return the array object corresponding to a nearby tile

    Args:
        r_delta: row offset from current agent
        c_delta: col offset from current agent

    Returns:
        Vector corresponding to the specified tile
    '''
    agent = self.agent()
    r_cond = (self.tiles[:,TileState.State.attr_name_to_col["row"]] == agent.row + r_delta)
    c_cond = (self.tiles[:,TileState.State.attr_name_to_col["col"]] == agent.col + c_delta)
    return TileState.parse_array(self.tiles[r_cond & c_cond][0])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def entity(self, entity_id):
    rows = self.entities.values[self.entities.ids == entity_id]
    if rows.size == 0:
      return None
    return EntityState.parse_array(rows[0])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def agent(self):
    return self.entity(self.agent_id)

  def to_gym(self):
    '''Convert the observation to a format that can be used by OpenAI Gym'''

    gym_obs = {
      "Tile": np.vstack([
        self.tiles,
        np.zeros((self.config.MAP_N_OBS - self.tiles.shape[0], self.tiles.shape[1]))
      ]),
      "Entity": np.vstack([
        self.entities.values, np.zeros((
          self.config.PLAYER_N_OBS - self.entities.values.shape[0],
          self.entities.values.shape[1]))
      ]),
    }

    if self.config.ITEM_SYSTEM_ENABLED:
      gym_obs["Inventory"] = np.vstack([
        self.inventory.values, np.zeros((
          self.config.INVENTORY_N_OBS - self.inventory.values.shape[0],
          self.inventory.values.shape[1]))
      ])

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      gym_obs["Market"] = np.vstack([
        self.market.values, np.zeros((
          self.config.MARKET_N_OBS - self.market.values.shape[0],
          self.market.values.shape[1]))
      ])

    gym_obs["ActionTargets"] = self.generate_action_targets()

    return gym_obs

  def generate_action_targets(self):
    # TODO(kywch): return all-0 masks for buy/sell/give during combat

    masks = {}
    masks[action.Move] = {
      action.Direction: self._generate_move_mask()
    }

    if self.config.COMBAT_SYSTEM_ENABLED:
      masks[action.Attack] = {
        action.Style: self._generate_allow_all_mask(action.Style.edges),
        action.Target: self._generate_attack_mask()
      }

    if self.config.ITEM_SYSTEM_ENABLED:
      masks[action.Use] = {
        action.InventoryItem: self._generate_use_mask()
      }

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      masks[action.Sell] = {
        action.InventoryItem: self._generate_sell_mask(),
        action.Price: None # allow any integer
      }
      masks[action.Buy] = {
        action.MarketItem: self._generate_buy_mask()
      }

    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      masks[action.Comm] = {
        action.Token: self._generate_allow_all_mask(action.Token.edges),
      }

    return masks

  def _generate_allow_all_mask(self, actions):
    return np.ones(len(actions), dtype=np.int8)

  def _generate_move_mask(self):
    # pylint: disable=not-an-iterable
    return np.array(
      [self.tile(*d.delta).material_id in material.Habitable
       for d in action.Direction.edges], dtype=np.int8)

  def _generate_attack_mask(self):
    # TODO: Currently, all attacks have the same range
    #   if we choose to make ranges different, the masks
    #   should be differently generated by attack styles
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_RANGE_REACH
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_MAGE_REACH
    assert self.config.COMBAT_RANGE_REACH == self.config.COMBAT_RANGE_REACH

    attack_range = self.config.COMBAT_MELEE_REACH

    agent = self.agent()
    dist_from_self = self.entities.dist((agent.row, agent.col))
    not_same_tile = dist_from_self > 0 # this also includes not_self
    within_range = dist_from_self <= attack_range

    if not self.config.COMBAT_FRIENDLY_FIRE:
      population = self.entities.values[:,EntityState.State.attr_name_to_col["population_id"]]
      no_friendly_fire = population != agent.population_id
    else:
      # allow friendly fire
      no_friendly_fire = np.ones(self.entities.len, dtype=np.int8)

    return np.concatenate([not_same_tile & within_range & no_friendly_fire,
      np.zeros(self.config.PLAYER_N_OBS - self.entities.len, dtype=np.int8)])

  def _generate_use_mask(self):
    # empty inventory -- nothing to use
    if self.inventory.len == 0:
      return np.zeros(self.config.INVENTORY_N_OBS, dtype=np.int8)

    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0
    item_type = self.inventory.values[:,ItemState.State.attr_name_to_col["type_id"]]
    item_level = self.inventory.values[:,ItemState.State.attr_name_to_col["level"]]

    # level limits are differently applied depending on item types
    type_flt = np.tile ( np.array(list(self._item_skill.keys())), (self.inventory.len,1) )
    level_flt = np.tile ( np.array(list(self._item_skill.values())), (self.inventory.len,1) )
    item_type = np.tile( np.transpose(np.atleast_2d(item_type)), (1, len(self._item_skill)))
    item_level = np.tile( np.transpose(np.atleast_2d(item_level)), (1, len(self._item_skill)))
    level_satisfied = np.any((item_type == type_flt) & (item_level <= level_flt), axis=1)

    return np.concatenate([not_listed & level_satisfied,
      np.zeros(self.config.INVENTORY_N_OBS - self.inventory.len, dtype=np.int8)])

  @property
  def _item_skill(self):
    agent = self.agent()

    # the minimum agent level is 1
    level = max(1, agent.melee_level, agent.range_level, agent.mage_level,
                agent.fishing_level, agent.herbalism_level, agent.prospecting_level,
                agent.carving_level, agent.alchemy_level)
    return {
      item_system.Hat.ITEM_TYPE_ID: level,
      item_system.Top.ITEM_TYPE_ID: level,
      item_system.Bottom.ITEM_TYPE_ID: level,
      item_system.Sword.ITEM_TYPE_ID: agent.melee_level,
      item_system.Bow.ITEM_TYPE_ID: agent.range_level,
      item_system.Wand.ITEM_TYPE_ID: agent.mage_level,
      item_system.Rod.ITEM_TYPE_ID: agent.fishing_level,
      item_system.Gloves.ITEM_TYPE_ID: agent.herbalism_level,
      item_system.Pickaxe.ITEM_TYPE_ID: agent.prospecting_level,
      item_system.Chisel.ITEM_TYPE_ID: agent.carving_level,
      item_system.Arcane.ITEM_TYPE_ID: agent.alchemy_level,
      item_system.Scrap.ITEM_TYPE_ID: agent.melee_level,
      item_system.Shaving.ITEM_TYPE_ID: agent.range_level,
      item_system.Shard.ITEM_TYPE_ID: agent.mage_level,
      item_system.Ration.ITEM_TYPE_ID: level,
      item_system.Poultice.ITEM_TYPE_ID: level
    }

  def _generate_sell_mask(self):
    # empty inventory -- nothing to sell
    if self.inventory.len == 0:
      return np.zeros(self.config.INVENTORY_N_OBS, dtype=np.int8)

    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0

    return np.concatenate([not_equipped & not_listed,
      np.zeros(self.config.INVENTORY_N_OBS - self.inventory.len, dtype=np.int8)])

  def _generate_buy_mask(self):
    market_flt = np.ones(self.market.len, dtype=np.int8)
    full_inventory = self.inventory.len >= self.config.ITEM_INVENTORY_CAPACITY

    # if the inventory is full, one can only buy existing ammo stack
    if full_inventory:
      exist_ammo_listings = self._existing_ammo_listings()
      if not np.any(exist_ammo_listings):
        return np.zeros(self.config.MARKET_N_OBS, dtype=np.int8)
      market_flt = exist_ammo_listings

    agent = self.agent()
    market_items = self.market.values
    enough_gold = market_items[:,ItemState.State.attr_name_to_col["listed_price"]] <= agent.gold
    not_mine = market_items[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id
    not_equipped = market_items[:,ItemState.State.attr_name_to_col["equipped"]] == 0

    return np.concatenate([market_flt & enough_gold & not_mine & not_equipped,
      np.zeros(self.config.MARKET_N_OBS - self.market.len, dtype=np.int8)])

  def _existing_ammo_listings(self):
    sig_col = (ItemState.State.attr_name_to_col["type_id"],
               ItemState.State.attr_name_to_col["level"])
    ammo_id = [ammo.ITEM_TYPE_ID for ammo in
              [item_system.Scrap, item_system.Shaving, item_system.Shard]]

    # search ammo stack from the inventory
    type_flt = np.tile( np.array(ammo_id), (self.inventory.len,1))
    item_type = np.tile(
      np.transpose(np.atleast_2d(self.inventory.values[:,sig_col[0]])),
      (1, len(ammo_id)))
    exist_ammo = self.inventory.values[np.any(item_type == type_flt, axis=1)]

    # self does not have ammo
    if exist_ammo.shape[0] == 0:
      return np.zeros(self.market.len, dtype=np.int8)

    # search the existing ammo stack from the market
    type_flt = np.tile( np.array(exist_ammo[:,sig_col[0]]), (self.market.len,1))
    level_flt = np.tile( np.array(exist_ammo[:,sig_col[1]]), (self.market.len,1))
    item_type = np.tile( np.transpose(np.atleast_2d(self.market.values[:,sig_col[0]])),
      (1, exist_ammo.shape[0]))
    item_level = np.tile( np.transpose(np.atleast_2d(self.market.values[:,sig_col[1]])),
      (1, exist_ammo.shape[0]))
    exist_ammo_listings = np.any((item_type == type_flt) & (item_level == level_flt), axis=1)

    not_mine = self.market.values[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id

    return exist_ammo_listings & not_mine
