#Various utilities for managing combat, including hit/damage

from pdb import set_trace as T

import numpy as np
from neural_mmo.forge.blade.systems import skill as Skill
from neural_mmo.forge.blade import item as Item

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
   skill_type = type(skill)

   #Base damage
   base    = config.DAMAGE_BASE

   #Weapon mod
   held      = entity.equipment.held
   held_type = type(held)
   weapon    = 0
   if skill_type == Skill.Melee and held_type != Item.Sword:
       pass
   elif skill_type == Skill.Range and held_type != Item.Bow:
       pass
   elif skill_type == Skill.Mage and held_type != Item.Wand:
       pass
   else:
       weapon  = entity.equipment.offense

   #Ammo mod
   ammo = 0
   if entity.equipment.ammunition:
      ammo = entity.equipment.ammunition.use(type(skill))

   #Style dominance multiplier
   mul     = damage_multiplier(config, skill, targ)

   #Attack and defense scores
   attack  = base + weapon + ammo
   defense = targ.equipment.defense

   #Total damage calculation
   dmg     = int(mul * (attack - defense))
   dmg     = min(dmg, entity.resources.health.val)

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
