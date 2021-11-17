from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.blade.io import node

def bind(gameCls):
   def to(ioCls):
      @property
      def GAME_CLS():
         return gameCls

      ioCls.GAME_CLS = GAME_CLS
      #ioCls.GAME_CLS = gameCls
      return ioCls
   return to 

class Config(metaclass=node.IterableNameComparable):
   pass

class Stimulus(Config):
   def dict():
      return {k[0] : v for k, v in dict(Stimulus).items()}

   class Entity(Config):
      @staticmethod
      def N(config):
         return config.N_AGENT_OBS

      class Self(node.Discrete):
         def init(self, config):
            self.max = 1
            self.scale = 1.0

      class ID(node.Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class AttackerID(node.Continuous):
         def init(self, config):
            self.min   = -np.inf
            self.scale = 0.001

      class Level(node.Continuous):
         def init(self, config):
            self.scale = 0.1

      class ItemLevel(node.Continuous):
         def init(self, config):
            self.scale = 0.025
            self.max   = 5 * config.NPC_LEVEL_MAX

      class Population(node.Discrete):
         def init(self, config):
            self.min = -3 #NPC index
            self.max = config.NPOP - 1
            self.scale = 1.0

      class R(node.Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

      class C(node.Discrete):
         def init(self, config):
            self.min = 0
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

      #Historical stats
      class Damage(node.Continuous):
         def init(self, config):
            #This scale may eventually be too high
            self.val   = 0
            self.scale = 0.1

      class TimeAlive(node.Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      #Status effects
      class Freeze(node.Continuous):
         def init(self, config):
            self.val = 0
            self.max = 3
            self.scale = 0.3

      class Gold(node.Continuous):
         def init(self, config):
            self.val = 0
            self.scale = 0.01

      #Resources -- Redo the max/min scaling. You can't change these
      #after init without messing up the embeddings
      class Health(node.Continuous):
         def init(self, config):
            self.val = config.BASE_HEALTH
            self.max = config.BASE_HEALTH
            self.scale = 0.1

      class Food(node.Continuous):
         def init(self, config):
            if config.game_system_enabled('Resource'):
               self.val = config.RESOURCE_BASE
               self.max = config.RESOURCE_BASE
            else:
               self.val = 1
               self.max = 1
    
            self.scale = 0.1

      class Water(node.Continuous):
         def init(self, config):
            if config.game_system_enabled('Resource'):
               self.val = config.RESOURCE_BASE
               self.max = config.RESOURCE_BASE
            else:
               self.val = 1
               self.max = 1
 
            self.scale = 0.1

      class Melee(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Range(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Mage(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Fishing(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Herbalism(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Prospecting(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Carving(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

      class Alchemy(node.Continuous):
         def init(self, config):
            self.val = 1
            self.max = 1
            if config.game_system_enabled('Progression'):
               self.max = config.PROGRESSION_LEVEL_MAX

   class Tile(Config):
      @staticmethod
      def N(config):
         return config.WINDOW**2

      class NEnts(node.Continuous):
         def init(self, config):
            self.max = config.NENT
            self.val = 0
            self.scale = 1.0

      class Index(node.Discrete):
         def init(self, config):
            self.max = config.NTILE
            self.scale = 0.15

      class R(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15
 
      class C(node.Discrete):
         def init(self, config):
            self.max = config.TERRAIN_SIZE - 1
            self.scale = 0.15

   class Item(Config):
      @staticmethod
      def N(config):
         return config.N_INVENTORY

      class ID(node.Continuous):
         def init(self, config):
            self.scale = 0.001

      class Index(node.Discrete):
         def init(self, config):
            self.max   = config.N_ITEM + 1
            self.scale = 1.0 / self.max

      class Level(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Capacity(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Quantity(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Tradable(node.Discrete):
         def init(self, config):
            self.max   = 1
            self.scale = 1.0

      class MeleeAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MeleeDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class HealthRestore(node.Continuous):
         def init(self, config):
            self.max   = 100 
            self.scale = 1.0 / self.max

      class ResourceRestore(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class Price(node.Continuous):
         def init(self, config):
            self.scale = 0.01

      class Equipped(node.Discrete):
         def init(self, config):
            self.scale = 1.0


   # ToDo: figure out how to autogen this
   class Market(Config):
      @staticmethod
      def N(config):
         return config.N_MARKET

      class ID(node.Continuous):
         def init(self, config):
            self.scale = 0.001

      class Index(node.Discrete):
         def init(self, config):
            self.max   = config.N_ITEM + 1
            self.scale = 1.0 / self.max

      class Level(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Capacity(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Quantity(node.Continuous):
         def init(self, config):
            self.max   = 99
            self.scale = 1.0 / self.max

      class Tradable(node.Discrete):
         def init(self, config):
            self.max   = 1
            self.scale = 1.0

      class MeleeAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageAttack(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MeleeDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class RangeDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class MageDefense(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class HealthRestore(node.Continuous):
         def init(self, config):
            self.max   = 100 
            self.scale = 1.0 / self.max

      class ResourceRestore(node.Continuous):
         def init(self, config):
            self.max   = 100
            self.scale = 1.0 / self.max

      class Price(node.Continuous):
         def init(self, config):
            self.scale = 0.01

      class Equipped(node.Discrete):
         def init(self, config):
            self.scale = 1.0


for objName, obj in Stimulus:
   for idx, (attrName, attr) in enumerate(obj):
      attr.index = idx 
