from functools import lru_cache
from types import SimpleNamespace
from nmmo.core.tile import TileState

from nmmo.entity.entity import EntityState
import numpy as np

from nmmo.systems.item import ItemState
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
    self.entities = SimpleNamespace(
        values = entities,
        ids = entities[:,EntityState._attr_name_to_col["id"]])

    if config.ITEM_SYSTEM_ENABLED:
      inventory = inventory[0:config.ITEM_N_OBS]
      self.inventory = SimpleNamespace(
        values = inventory,
        ids = inventory[:,ItemState._attr_name_to_col["id"]])
    else:
      assert inventory.size == 0

    if config.EXCHANGE_SYSTEM_ENABLED:
      market = market[0:config.EXCHANGE_N_OBS]
      self.market = SimpleNamespace(
        values = market,
        ids = market[:,ItemState._attr_name_to_col["id"]])
    else:
      assert market.size == 0

  @lru_cache(maxsize=None)
  def tile(self, rDelta, cDelta):
    '''Return the array object corresponding to a nearby tile
    
    Args:
        rDelta: row offset from current agent
        cDelta: col offset from current agent

    Returns:
        Vector corresponding to the specified tile
    '''
    agent = self.agent()
    r_cond = (self.tiles[:,TileState._attr_name_to_col["r"]] == agent.r + rDelta)
    c_cond = (self.tiles[:,TileState._attr_name_to_col["c"]] == agent.c + cDelta)
    return TileState.parse_array(self.tiles[r_cond & c_cond][0])

  @lru_cache(maxsize=None)
  def entity(self, entity_id):
    rows = self.entities.values[self.entities.ids == entity_id]
    if rows.size == 0:
      return None
    return EntityState.parse_array(rows[0])

  @lru_cache(maxsize=None)
  def agent(self):
    return self.entity(self.agent_id)

  def to_gym(self):
    '''Convert the observation to a format that can be used by OpenAI Gym'''

    # TODO: The padding slows things down significantly.
    # maybe there's a better way?
        
    # gym_obs = {
    #   "Tile": self.tiles, 
    #   "Entity": self.entities.values, 
    # }
    # if self.config.ITEM_SYSTEM_ENABLED:
    #   gym_obs["Inventory"] = self.inventory.values
    
    # if self.config.EXCHANGE_SYSTEM_ENABLED:
    #   gym_obs["Market"] = self.market.values
    # return gym_obs
    
    gym_obs = {
      "Tile": np.pad(
        self.tiles, 
        [(0, self.config.MAP_N_OBS - self.tiles.shape[0]), (0, 0)],
        mode="constant"),

      "Entity": np.pad(
        self.entities.values, 
        [(0, self.config.PLAYER_N_OBS - self.entities.values.shape[0]), (0, 0)],
        mode="constant")
    }
    
    if self.config.ITEM_SYSTEM_ENABLED:
      gym_obs["Inventory"] = np.pad(
        self.inventory.values,
        [(0, self.config.ITEM_N_OBS - self.inventory.values.shape[0]), (0, 0)],
        mode="constant")
    
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      gym_obs["Market"] = np.pad(
        self.market.values,
        [(0, self.config.EXCHANGE_N_OBS - self.market.values.shape[0]), (0, 0)],
        mode="constant")

    return gym_obs