from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.blade.systems import skill, droptable, combat, inventory, equipment
from neural_mmo.forge.blade.lib import material, utils

from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.io.stimulus import Static

class Resources:
   def __init__(self, ent):
      self.health = Static.Entity.Health(ent.dataframe, ent.entID)
      self.water  = Static.Entity.Water( ent.dataframe, ent.entID)
      self.food   = Static.Entity.Food(  ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      config      = realm.config

      self.water.max  = config.RESOURCE_BASE
      self.food.max   = config.RESOURCE_BASE

      regen       = config.RESOURCE_HEALTH_RESTORE_FRACTION
      thresh      = config.RESOURCE_HEALTH_REGEN_THRESHOLD

      foodThresh  = self.food  > thresh * config.RESOURCE_BASE
      waterThresh = self.water > thresh * config.RESOURCE_BASE

      if foodThresh and waterThresh:
         restore = np.floor(self.health.max * regen)
         self.health.increment(restore)

      if self.food.empty:
         self.health.decrement(config.RESOURCE_STARVATION_RATE)

      if self.water.empty:
         self.health.decrement(config.RESOURCE_DEHYDRATION_RATE)

   def packet(self):
      data = {}
      data['health'] = self.health.packet()
      data['food']   = self.food.packet()
      data['water']  = self.water.packet()
      return data

class Status:
   def __init__(self, ent):
      self.config = ent.config
      self.freeze     = Static.Entity.Freeze(ent.dataframe, ent.entID)

   def update(self, realm, entity, actions):
      self.freeze.decrement()

   def packet(self):
      data = {}
      data['freeze']     = self.freeze.val
      return data

class History:
   def __init__(self, ent):
      self.actions = None
      self.attack  = None
  
      self.origPos     = ent.pos
      self.exploration = 0
      self.playerKills = 0

      self.damage    = Static.Entity.Damage(   ent.dataframe, ent.entID)
      self.timeAlive = Static.Entity.TimeAlive(ent.dataframe, ent.entID)

      self.lastPos = None

   def update(self, realm, entity, actions):
      self.attack  = None
      self.actions = actions
      self.damage.update(0)

      exploration      = utils.linf(entity.pos, self.origPos)
      self.exploration = max(exploration, self.exploration)

      self.timeAlive.increment()

   def packet(self):
      data = {}
      data['damage']    = self.damage.val
      data['timeAlive'] = self.timeAlive.val

      if self.attack is not None:
         data['attack'] = self.attack

      return data

class Base:
   def __init__(self, ent, pos, iden, name, color, pop):
      self.name  = '{}_{}'.format(name, iden)
      self.color = color
      r, c       = pos

      self.r          = Static.Entity.R(ent.dataframe, ent.entID, r)
      self.c          = Static.Entity.C(ent.dataframe, ent.entID, c)

      self.population = Static.Entity.Population(ent.dataframe, ent.entID, pop)
      self.self       = Static.Entity.Self(      ent.dataframe, ent.entID, 1)
      self.identity   = Static.Entity.ID(        ent.dataframe, ent.entID, ent.entID)
      self.level      = Static.Entity.Level(     ent.dataframe, ent.entID, 1)
      self.item_level = Static.Entity.ItemLevel( ent.dataframe, ent.entID, 0)
      self.gold       = Static.Entity.Gold(      ent.dataframe, ent.entID, 0)

      ent.dataframe.init(Static.Entity, ent.entID, (r, c))

   def update(self, realm, entity, actions):
      self.level.update(combat.level(entity.skills))
      self.item_level.update(entity.equipment.total(lambda e: e.level))
      self.gold.update(entity.inventory.gold.quantity.val)

   @property
   def pos(self):
      return self.r.val, self.c.val

   def packet(self):
      data = {}

      data['r']          = self.r.val
      data['c']          = self.c.val
      data['name']       = self.name
      data['level']      = self.level.val
      data['item_level'] = self.item_level.val
      data['color']      = self.color.packet()
      data['population'] = self.population.val
      data['self']       = self.self.val

      return data

class Entity:
   def __init__(self, realm, pos, iden, name, color, pop):
      self.dataframe    = realm.dataframe
      self.config       = realm.config

      self.policy       = name
      self.entID        = iden
      self.repr         = None
      self.vision       = 5

      self.attacker     = None
      self.target       = None
      self.closest      = None
      self.spawnPos     = pos

      self.attackerID = Static.Entity.AttackerID(self.dataframe, self.entID, 0)

      #Submodules
      self.base      = Base(self, pos, iden, name, color, pop)
      self.status    = Status(self)
      self.history   = History(self)
      self.resources = Resources(self)
      self.inventory = inventory.Inventory(realm, self)

   def packet(self):
      data = {}

      data['status']    = self.status.packet()
      data['history']   = self.history.packet()
      data['inventory'] = self.inventory.packet()
      data['alive']     = self.alive

      return data

   def update(self, realm, actions):
      '''Update occurs after actions, e.g. does not include history'''
      if self.history.damage == 0:
         self.attacker = None
         self.attackerID.update(0)

      self.base.update(realm, self, actions)
      self.status.update(realm, self, actions)
      self.history.update(realm, self, actions)

   def receiveDamage(self, source, dmg):
      self.history.damage.update(dmg)
      self.resources.health.decrement(dmg)

      if not self.alive and source is not None:
         for item in list(self.inventory._items):
            self.inventory.remove(item)
            if source.inventory.space:
               source.inventory.receive(item)
         return False

      return True

   def applyDamage(self, dmg, style):
      pass

   @property
   def pos(self):
      return self.base.pos

   @property
   def alive(self):
      if self.resources.health.empty:
         return False

      return True

   @property
   def isPlayer(self) -> bool:
      return False

   @property
   def isNPC(self) -> bool:
      return False

   @property
   def level(self) -> int:
      melee  = self.skills.melee.level.val
      ranged = self.skills.range.level.val
      mage   = self.skills.mage.level.val

      return int(max(melee, ranged, mage))

   @property
   def equipment(self):
      return self.inventory.equipment
