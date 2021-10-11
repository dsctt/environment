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

def damage_multiplier(config, skill, targ):
   skills = [targ.skills.melee, targ.skills.range, targ.skills.mage]
   idx    = np.argmax([s.level for s in skills])
   targ   = skills[idx]

   if type(targ) == skill.weakness:
       return config.DAMAGE_MULTIPLIER 

   return 1.0

def attack(entity, targ, skillFn):
   config = entity.config
   skill  = skillFn(entity)

   #Base damage
   base    = config.DAMAGE_BASE

   #Weapon mod
   weapon  = entity.inventory.equipment.offense

   #Ammo mod
   ammo = entity.inventory.equipment.use_ammunition(type(skill))

   #Style dominance multiplier
   mul     = damage_multiplier(config, skill, targ)

   #Attack and defense scores
   attack  = base# + weapon + ammo
   defense = entity.inventory.equipment.defense

   #Total damage calculation
   dmg     = mul * (attack - defense)
   dmg     = max(int(dmg), 0)
   dmg     = min(int(dmg), entity.resources.health.val)

   entity.applyDamage(dmg, skill.__class__.__name__.lower())
   targ.receiveDamage(entity, dmg)

   return dmg

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
