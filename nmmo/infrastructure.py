'''Infrastructure layer for representing agent observations

Maintains a synchronized + serialized representation of agent observations in
flat tensors. This allows for fast observation processing as a set of tensor
slices instead of a lengthy traversal over hundreds of game properties.

Synchronization bugs are notoriously difficult to track down: make sure
to follow the correct instantiation protocol, e.g. as used for defining
agent/tile observations, when adding new types observations to the code'''

from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import nmmo

class DataType:
   CONTINUOUS = np.float32
   DISCRETE   = np.int32

class Index:
   '''Lookup index of attribute names'''
   def __init__(self, prealloc):
      # Key 0 is reserved as padding
      self.free  = {idx for idx in range(1, prealloc)}
      self.index = {}
      self.back  = {}

   def full(self):
      return len(self.free) == 0

   def remove(self, key):
      row = self.index[key]
      del self.index[key]
      del self.back[row]

      self.free.add(row)
      return row

   def update(self, key):
      if key in self.index:
         row = self.index[key]
      else:
         row = self.free.pop()
         self.index[key] = row
         self.back[row]  = key

      return row

   def get(self, key):
      return self.index[key]

   def teg(self, row):
      return self.back[row]

   def expand(self, cur, nxt):
      self.free.update({idx for idx in range(cur, nxt)})

class ContinuousTable:
   '''Flat tensor representation for a set of continuous attributes'''
   def __init__(self, config, obj, prealloc, dtype=DataType.CONTINUOUS):
      self.config = config
      self.dtype  = dtype
      self.cols   = {}
      self.nCols  = 0

      for (attribute,), attr in obj:
         self.initAttr(attribute, attr)

      self.data = self.initData(prealloc, self.nCols)

   def initAttr(self, key, attr):
      if attr.CONTINUOUS:
         self.cols[key] = self.nCols
         self.nCols += 1

   def initData(self, nRows, nCols):
      return np.zeros((nRows, nCols), dtype=self.dtype)

   def update(self, row, attr, val):
      col = self.cols[attr] 
      self.data[row, col] = val

   def expand(self, cur, nxt):
      data       = self.initData(nxt, self.nCols)
      data[:cur] = self.data

      self.data  = data
      self.nRows = nxt

   def get(self, rows, pad=None):
      data = self.data[rows]

      # This call is expensive
      # Padding index 0 should make this redundant
      # data[rows==0] = 0

      if pad is not None:
         data = np.pad(data, ((0, pad-len(data)), (0, 0)))

      return data

class DiscreteTable(ContinuousTable):
   '''Flat tensor representation for a set of discrete attributes'''
   def __init__(self, config, obj, prealloc, dtype=DataType.DISCRETE):
      self.discrete, self.cumsum = {}, 0
      super().__init__(config, obj, prealloc, dtype)

   def initAttr(self, key, attr):
      if not attr.DISCRETE:
         return

      self.cols[key]     =  self.nCols

      #Flat index
      attr               =  attr(None, None, 0, config=self.config)
      self.discrete[key] =  self.cumsum

      self.cumsum        += attr.max - attr.min + 1
      self.nCols         += 1

   def update(self, row, attr, val):
      col = self.cols[attr] 
      self.data[row, col] = val + self.discrete[attr]

class Grid:
   '''Flat representation of tile/agent positions'''
   def __init__(self, R, C):
      self.data = np.zeros((R, C), dtype=np.int32)

   def zero(self, pos):
      r, c            = pos
      self.data[r, c] = 0
    
   def set(self, pos, val):
      r, c            = pos
      self.data[r, c] = val
 
   def move(self, pos, nxt, row):
      self.zero(pos)
      self.set(nxt, row)

   def window(self, rStart, rEnd, cStart, cEnd):
      crop = self.data[rStart:rEnd, cStart:cEnd].ravel()
      return list(filter(lambda x: x != 0, crop))
      
class GridTables:
   '''Combines a Grid + Index + Continuous and Discrete tables

   Together, these data structures provide a robust and efficient
   flat tensor representation of an entire class of observations,
   such as agents or tiles'''
   def __init__(self, config, obj, pad, prealloc=1000, expansion=2):
      self.grid       = Grid(config.TERRAIN_SIZE, config.TERRAIN_SIZE)
      self.continuous = ContinuousTable(config, obj, prealloc)
      self.discrete   = DiscreteTable(config, obj, prealloc)
      self.index      = Index(prealloc)

      self.nRows      = prealloc
      self.expansion  = expansion
      self.radius     = config.NSTIM
      self.pad        = pad

   def update(self, obj, val):
      key, attr = obj.key, obj.attr
      if self.index.full():
         cur        = self.nRows
         self.nRows = cur * self.expansion

         self.index.expand(cur, self.nRows)
         self.continuous.expand(cur, self.nRows)
         self.discrete.expand(cur, self.nRows)

      row = self.index.update(key)
      if obj.DISCRETE:
         self.discrete.update(row, attr, val - obj.min)
      if obj.CONTINUOUS:
         self.continuous.update(row, attr, val)

   def move(self, key, pos, nxt):
      row = self.index.get(key)
      self.grid.move(pos, nxt, row)

   def init(self, key, pos):
      row = self.index.get(key)
      self.grid.set(pos, row)

   def remove(self, key, pos):
      self.index.remove(key)
      self.grid.zero(pos)

class Dataframe:
   '''Infrastructure wrapper class'''
   def __init__(self, config):
      self.config, self.data = config, defaultdict(dict)

      for (objKey,), obj in nmmo.Serialized:
         self.data[objKey] = GridTables(config, obj, pad=obj.N(config))

      # Preallocate index buffers
      radius = config.NSTIM
      self.N = int(config.WINDOW ** 2)
      cent = self.N // 2

      rr, cc = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
      rr, cc = rr.ravel(), cc.ravel()
      rr = np.repeat(rr[None, :], config.NENT, axis=0)
      cc = np.repeat(cc[None, :], config.NENT, axis=0)
      self.tile_grid = (rr, cc)

      rr, cc = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
      rr, cc = rr.ravel(), cc.ravel()
      rr[0], rr[cent] = rr[cent], rr[0]
      cc[0], cc[cent] = cc[cent], cc[0]
      rr = np.repeat(rr[None, :], config.NENT, axis=0)
      cc = np.repeat(cc[None, :], config.NENT, axis=0)
      self.player_grid = (rr, cc)

   def update(self, node, val):
      self.data[node.obj].update(node, val)

   def remove(self, obj, key, pos):
      self.data[obj.__name__].remove(key, pos)

   def init(self, obj, key, pos):
      self.data[obj.__name__].init(key, pos)

   def move(self, obj, key, pos, nxt):
      self.data[obj.__name__].move(key, pos, nxt)

   def get(self, players):
      obs, action_lookup = {}, {}

      n = len(players)
      r_offsets = np.zeros((n, 1), dtype=int)
      c_offsets = np.zeros((n, 1), dtype=int)
      for idx, (playerID, player) in enumerate(players.items()):
          obs[playerID] = {}
          action_lookup[playerID] = {}
          
          r, c = player.pos
          r_offsets[idx] = r
          c_offsets[idx] = c

      for key, (rr, cc) in (('Entity', self.player_grid), ('Tile', self.tile_grid)):
          data = self.data[key]

          #TODO: Optimize this line with flat dataframes + np.take or ranges
          try:
            dat = data.grid.data[rr[:n] + r_offsets, cc[:n] + c_offsets]#.ravel()
          except:
            T()
          key_mask = dat != 0

          # TODO: Optimize these two lines with some sort of jit... it's a dict lookup
          continuous = data.continuous.get(dat, None)
          discrete = data.discrete.get(dat, None)

          for idx, (playerID, _) in enumerate(players.items()):
              obs[playerID][key] = {
                      'Continuous': continuous[idx],
                      'Discrete': discrete[idx],
                      'Mask': key_mask[idx]}

              action_lookup[playerID][key] = dat[idx]

      return obs, action_lookup
