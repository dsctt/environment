from pdb import set_trace as T
import numpy as np

class Observation:
   '''Unwraps observation tensors for use with scripted agents'''
   def __init__(self, config, obs):
      '''
      Args:
          config: A forge.blade.core.Config object or subclass object
          obs: An observation object from the environment
      '''
      self.config = config
      self.obs    = obs
      self.delta  = config.PLAYER_VISION_RADIUS
      self.tiles  = self.obs['Tile']['Continuous']
      self.agents = self.obs['Entity']['Continuous']

      agents = self.obs['Entity']
      self.agents = agents['Continuous'][agents['Mask']]
      self.agent_mask_map = np.where(agents['Mask'])[0]

      if config.ITEM_SYSTEM_ENABLED:
          items = self.obs['Item']
          self.items = items['Continuous'][items['Mask']]
          self.items_mask_map = np.where(items['Mask'])[0]

      if config.EXCHANGE_SYSTEM_ENABLED:
          market = self.obs['Market']
          self.market = market['Continuous'][market['Mask']]
          self.market_mask_map = np.where(market['Mask'])[0]

   def tile(self, rDelta, cDelta):
      '''Return the array object corresponding to a nearby tile
      
      Args:
         rDelta: row offset from current agent
         cDelta: col offset from current agent

      Returns:
         Vector corresponding to the specified tile
      '''
      return self.tiles[self.config.PLAYER_VISION_DIAMETER * (self.delta + cDelta) + self.delta + rDelta]

   @property
   def agent(self):
      '''Return the array object corresponding to the current agent'''
      curr_idx = (self.config.PLAYER_VISION_DIAMETER + 1) * self.delta
      return self.obs['Entity']['Continuous'][curr_idx]

   @staticmethod
   def attribute(ary, attr):
      '''Return an attribute of a game object

      Args:
         ary: The array corresponding to a game object
         attr: A forge.blade.io.stimulus.static stimulus class
      '''
      return float(ary[attr.index])
