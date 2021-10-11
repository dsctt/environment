from pdb import set_trace as T
import abc

import numpy as np
from neural_mmo.forge.blade.systems import experience, combat, ai

from neural_mmo.forge.blade.lib import material

### Infrastructure ###
class SkillGroup:
   def __init__(self, realm):
      self.expCalc = experience.ExperienceCalculator()
      self.config  = realm.dataframe.config
      self.skills  = set()

   def update(self, realm, entity, actions):
      for skill in self.skills:
         skill.update(realm, entity)

   def packet(self):
      data = {}
      for skill in self.skills:
         data[skill.__class__.__name__.lower()] = skill.packet()

      return data

class Skill:
   skillItems = abc.ABCMeta

   def __init__(self, skillGroup):
      self.config  = skillGroup.config
      self.expCalc = skillGroup.expCalc
      self.exp     = 0

      skillGroup.skills.add(self)

   def packet(self):
      data = {}

      data['exp']   = self.exp
      data['level'] = self.level

      return data

   def update(self, xp):
      scale     = self.config.XP_SCALE
      self.exp += scale * xp

   def setExpByLevel(self, level):
      self.exp = self.expCalc.expAtLevel(level)

   @property
   def level(self):
      lvl = self.expCalc.levelAtExp(self.exp)
      assert lvl == int(lvl)
      return int(lvl)

### Skill Bases ###
class CombatSkill(Skill):
   def update(self, realm, entity):
      pass

class NonCombatSkill(Skill):
   def success(self, levelReq):
      level = self.level
      if level < levelReq:
         return False
      chance = 0.5 + 0.05*(level-levelReq)
      if chance >= 1.0:
         return True
      return np.random.rand() < chance

   def attempt(self, inv, item):
      if (item.createSkill != self.__class__ or
            self.level < item.createLevel):
         return

      if item.recipe is not None:
         #Check that everything is available
         if not inv.satisfies(item.recipe): return
         inv.removeRecipe(item.recipe)

      if item.alwaysSucceeds or self.success(item.createLevel):
         inv.add(item, item.amtMade)
         self.exp += item.exp
         return True

class HarvestSkill(NonCombatSkill):
   def processDrops(self, realm, entity, dropTable):
      drops = dropTable.roll(realm, self.level)
      entity.inventory.receive(drops)
      self.exp += 10 * self.config.PROGRESSION_BASE_XP_SCALE
        
   def harvest(self, realm, entity, matl, deplete=True):
      r, c = entity.pos
      if realm.map.tiles[r, c].state != matl:
         return

      if dropTable := realm.map.harvest(r, c, deplete):
         self.processDrops(realm, entity, dropTable)
         return True

   def harvestAdjacent(self, realm, entity, matl, deplete=True):
      r, c      = entity.pos
      dropTable = None

      if realm.map.tiles[r-1, c].state == matl:
         dropTable = realm.map.harvest(r-1, c, deplete)
      if realm.map.tiles[r+1, c].state == matl:
         dropTable = realm.map.harvest(r+1, c, deplete)
      if realm.map.tiles[r, c-1].state == matl:
         dropTable = realm.map.harvest(r, c-1, deplete)
      if realm.map.tiles[r, c+1].state == matl:
         dropTable = realm.map.harvest(r, c+1, deplete)

      if dropTable:
         self.processDrops(realm, entity, dropTable)
         return True

### Skill groups ###
class Basic(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.water = Water(self)
      self.food  = Food(self)

   @property
   def basicLevel(self):
      return 0.5 * (self.water.level 
                  + self.food.level)

class Harvest(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.fishing      = Fishing(self)
      self.hunting      = Hunting(self)
      self.prospecting  = Prospecting(self)
      self.carving      = Carving(self)
      self.alchemy      = Alchemy(self)

   @property
   def harvestLevel(self):
      return max(self.fishing.level,
                 self.hunting.level,
                 self.prospecting.level,
                 self.carving.level,
                 self.alchemy.level)

class Combat(SkillGroup):
   def __init__(self, realm):
      super().__init__(realm)

      self.melee        = Melee(self)
      self.range        = Range(self)
      self.mage         = Mage(self)

   def packet(self):
      data          = super().packet() 
      data['level'] = combat.level(self)

      return data

   @property
   def combatLevel(self):
      return max(self.melee.level,
                 self.range.level,
                 self.mage.level)

   def applyDamage(self, dmg, style):
      if not self.config.game_system_enabled('Progression'):
         return

      config     = self.config
      scale      = config.PROGRESSION_BASE_XP_SCALE

      skill      = self.__dict__[style]
      skill.exp += scale * dmg * config.PROGRESSION_COMBAT_XP_SCALE

   def receiveDamage(self, dmg):
      pass

class Skills(Basic, Harvest, Combat):
   pass

### Skills ###
class Melee(CombatSkill): pass
class Range(CombatSkill): pass
class Mage(CombatSkill): pass

Melee.weakness = Mage
Range.weakness = Melee
Mage.weakness  = Range

### Individual Skills ###
class CombatSkill(Skill): pass

class Constitution(CombatSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.BASE_HEALTH)

   def update(self, realm, entity):
      health = entity.resources.health
      food   = entity.resources.food
      water  = entity.resources.water
      config = self.config

      if not config.game_system_enabled('Resource'):
         health.increment(1)
         return

      # Heal if above fractional resource threshold
      regen       = config.RESOURCE_HEALTH_REGEN_THRESHOLD
      foodThresh  = food > regen * entity.skills.hunting.level
      waterThresh = water > regen * entity.skills.fishing.level

      if foodThresh and waterThresh:
         restore = config.RESOURCE_HEALTH_RESTORE_FRACTION
         restore = np.floor(restore * self.level)
         health.increment(restore)

      if food.empty:
         health.decrement(1)

      if water.empty:
         health.decrement(1)

class Water(HarvestSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      config, level = self.config, 1
      if config.game_system_enabled('Progression'):
         level = config.PROGRESSION_BASE_RESOURCE
      elif config.game_system_enabled('Resource'):
         level = config.RESOURCE_BASE_RESOURCE

      self.setExpByLevel(level)

   def update(self, realm, entity):
      if not self.config.game_system_enabled('Resource'):
         return

      water = entity.resources.water
      water.decrement(1)

      tiles = realm.map.tiles
      if not self.harvestAdjacent(realm, entity, material.Water, deplete=False):
         return

      restore = self.config.RESOURCE_HARVEST_RESTORE_FRACTION
      restore = np.floor(restore * self.level)
      water.increment(restore)

class Food(HarvestSkill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      config, level = self.config, 1
      if config.game_system_enabled('Progression'):
         level = config.PROGRESSION_BASE_RESOURCE
      elif config.game_system_enabled('Resource'):
         level = config.RESOURCE_BASE_RESOURCE

      self.setExpByLevel(level)

   def update(self, realm, entity):
      if not self.config.game_system_enabled('Resource'):
         return

      food = entity.resources.food
      food.decrement(1)

      if not self.harvest(realm, entity, material.Forest):
         return

      restore = self.config.RESOURCE_HARVEST_RESTORE_FRACTION
      restore = np.floor(restore * self.level)
      food.increment(restore)

class Fishing(HarvestSkill):
   def update(self, realm, entity):
      self.harvestAdjacent(realm, entity, material.Fish)

class Hunting(HarvestSkill):
   def update(self, realm, entity):
      self.harvest(realm, entity, material.Herb)

class Prospecting(HarvestSkill):
   def update(self, realm, entity):
      self.harvest(realm, entity, material.Ore)

class Carving(HarvestSkill):
   def update(self, realm, entity):
      self.harvest(realm, entity, material.Tree)

class Alchemy(HarvestSkill):
   def update(self, realm, entity):
      self.harvest(realm, entity, material.Crystal)

