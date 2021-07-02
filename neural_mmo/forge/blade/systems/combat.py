#Various utilities for managing combat, including hit/damage

from pdb import set_trace as T

import numpy as np
from neural_mmo.forge.blade.systems import skill as Skill

def level(skills):
   melee   = skills.melee.level
   ranged  = skills.range.level
   mage    = skills.mage.level
   
   final = max(melee, ranged, mage)
   return final

def dev_combat(entity, targ, skillFn):
   config = entity.config
   skill  = skillFn(entity)
   dmg    = damageFn(config, skill.__class__)(skill.level)
   
   item = entity.inventory.ammunition.use(skill)
   if entity.isPlayer and item:
      dmg += item.damage
      
   dmg = min(dmg, entity.resources.health.val)
   entity.applyDamage(dmg, skill.__class__.__name__.lower())
   targ.receiveDamage(entity, dmg)
   return dmg

def attack(entity, targ, skillFn):
   config      = entity.config
   if config.DEV_COMBAT:
      return dev_combat(entity, targ, skillFn)

   entitySkill = skillFn(entity)
   targetSkill = skillFn(targ)

   targetDefense = targ.skills.defense.level + targ.loadout.defense

   die  = config.COMBAT_DICE_SIDES
   roll = np.random.randint(1, die+1)
   dc   = accuracy(config, entitySkill.level, targetSkill.level, targetDefense)
   crit = roll == die

   dmg = 1 #Chip dmg on a miss
   if roll >= dc or crit:
      dmg = damage(entitySkill.__class__, entitySkill.level)
      
   dmg = min(dmg, entity.resources.health.val)
   entity.applyDamage(dmg, entitySkill.__class__.__name__.lower())
   targ.receiveDamage(entity, dmg)
   return dmg

#Compute maximum damage roll
def damageFn(config, skill):
   if skill == Skill.Melee:
      return config.DAMAGE_MELEE
   if skill == Skill.Range:
      return config.DAMAGE_RANGE
   if skill == Skill.Mage:
      return config.DAMAGE_MAGE

#Compute maximum attack or defense roll (same formula)
#Max attack 198 - min def 1 = 197. Max 198 - max 198 = 0
#REMOVE FACTOR OF 2 FROM ATTACK AFTER IMPLEMENTING WEAPONS
def accuracy(config, entAtk, targAtk, targDef):
   alpha   = config.COMBAT_DEFENSE_WEIGHT

   attack  = entAtk
   defense = alpha*targDef + (1-alpha)*targAtk
   dc      = defense - attack + config.COMBAT_DICE_SIDES//2

   return dc

def danger(config, pos, full=False):
   border = config.TERRAIN_BORDER
   center = config.TERRAIN_CENTER
   r, c   = pos
  
   #Distance from border
   rDist  = min(r - border, center + border - r - 1)
   cDist  = min(c - border, center + border - c - 1)
   dist   = min(rDist, cDist)
   norm   = 2 * dist / center

   if full:
      return norm, mag

   return norm
