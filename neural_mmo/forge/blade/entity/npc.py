from pdb import set_trace as T
import numpy as np

import random

from neural_mmo.forge.blade import item as Item
from neural_mmo.forge.blade.entity import entity, Player
from neural_mmo.forge.blade.systems import combat, equipment, ai, combat, skill
from neural_mmo.forge.blade.lib.enums import Neon
from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.io.stimulus import Static

class NPC(entity.Entity):
   def __init__(self, realm, pos, iden, name, color, pop):
      super().__init__(realm, pos, iden, name, color, pop)
      self.skills = skill.Combat(self)

   def update(self, realm, actions):
      super().update(realm, actions)

      if not self.alive:
         return

      self.resources.health.increment(1)
      self.lastAction = actions

   @staticmethod
   def spawn(realm, pos, iden):
      config = realm.config

      #Select AI Policy
      danger           = combat.danger(config, pos)
      if danger >= config.NPC_SPAWN_AGGRESSIVE:
         ent = Aggressive(realm, pos, iden)
      elif danger >= config.NPC_SPAWN_NEUTRAL:
         ent = PassiveAggressive(realm, pos, iden)
      else:
         ent = Passive(realm, pos, iden)

      ent.spawn_danger = danger

      #Compute level
      level_min = config.NPC_LEVEL_MIN
      level_max = config.NPC_LEVEL_MAX
      level     = int(danger * (level_max - level_min) + level_min)

      #Select combat focus
      style = random.choice(
         (Action.Melee, Action.Range, Action.Mage))

      #Equipment to instantiate
      equipment =  [Item.Hat, Item.Top, Item.Bottom]
      tools     =  [Item.Rod, Item.Gloves, Item.Pickaxe, Item.Chisel, Item.Arcane]
      equipment.append(random.choice(tools))

      #Set skills
      ent.skills.style = style
      if style == Action.Melee:
         ent.skills.melee.setExpByLevel(level)
         equipment.append(Item.Sword)
      elif style == Action.Range:
         ent.skills.range.setExpByLevel(level)
         equipment.append(Item.Bow)
      elif style == Action.Mage:
         ent.skills.mage.setExpByLevel(level)
         equipment.append(Item.Wand)

      ent.resources.health.max = level
      ent.resources.health.update(level)

      #Select one piece of equipment to match the agent's level
      #The rest will be one tier lower
      upgrade = random.choice(equipment)
      for equip in equipment:
          if equip is upgrade:
              itm = equip(realm, level)
          elif level == 1:
              continue
          else:
              itm = equip(realm, level - 1)

          ent.inventory.receive(itm)
          if not isinstance(itm, Item.Tool):
             itm.use(ent)

      ent.inventory.gold.quantity.update(level)
      return ent

   def yieldDrops(self):
      self.lastAttacker.receiveDrops(self.drops.roll())

   @staticmethod
   def gearLevel(lvl, offset=10):
      proposed = random.gauss(lvl-offset, offset)
      lvl      = np.clip(proposed, 1, lvl)
      return int(lvl)

   @staticmethod
   def clippedLevels(config, danger, n=1):
      lmin    = config.NPC_LEVEL_MIN
      lmax    = config.NPC_LEVEL_MAX

      lbase   = danger*(lmax-lmin) + lmin
      lspread = config.NPC_LEVEL_SPREAD

      lvlMin  = int(max(lmin, lbase - lspread))
      lvlMax  = int(min(lmax, lbase + lspread))

      lvls = [random.randint(lvlMin, lvlMax) for _ in range(n)]

      if n == 1:
         return lvls[0]

      return lvls
 
   def packet(self):
      data = super().packet()

      data['base']     = self.base.packet()      
      data['skills']   = self.skills.packet()      
      data['resource'] = {'health': self.resources.health.packet()}

      return data

   @property
   def isNPC(self) -> bool:
      return True

class Passive(NPC):
   policy = 'Passive'
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, Passive.policy, Neon.GREEN, -1)
      self.dataframe.init(Static.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.passive(realm, self)

class PassiveAggressive(NPC):
   policy = 'Neutral'
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, PassiveAggressive.policy, Neon.ORANGE, -2)
      self.dataframe.init(Static.Entity, iden, pos)

   def decide(self, realm):
      return ai.policy.neutral(realm, self)

class Aggressive(NPC):
   policy = 'Hostile'
   def __init__(self, realm, pos, iden):
      super().__init__(realm, pos, iden, Aggressive.policy, Neon.RED, -3)
      self.dataframe.init(Static.Entity, iden, pos)
      self.vision = int(max(self.vision, 1 + combat.level(self.skills) // 10))
      self.dataframe.init(Static.Entity, self.entID, self.pos)

   def decide(self, realm):
      return ai.policy.hostile(realm, self)
