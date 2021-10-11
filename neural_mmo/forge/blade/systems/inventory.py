from pdb import set_trace as T
import numpy as np

import inspect

from neural_mmo.forge.blade import Item
from neural_mmo.forge.blade.systems import skill as Skill

class Pouch:
   def __init__(self, capacity):
      self.capacity = capacity
      self.items    = []

   @property
   def space(self):
      return self.capacity - len(self.items)

   @property
   def packet(self):
      return [item.packet for item in self.items]

   def __iter__(self):
      for item in self.items:
         yield item

   def add(self, item):
      space = self.space
      err = '{} out of space for {}'
      assert space, err.format(self, item) 
      self.items.append(item)

   def remove(self, item, level=None):
      itm =  self.get(item, level, remove=True)
      assert itm, 'item.remove: {} not in pouch {}'.format(item, type(self))
      return itm

   def get(self, item, level=None, remove=False):
      for itm in self.items:
         if type(itm) != item:
            continue

         if level is not None and level != item.level.val:
            continue

         if remove:
            self.items.remove(itm)

         return itm

   def use(self, item):
      item = self.get(item)

      if not items:
         return False

      if not item.use():
         return False

      return item

class Loadout:
   def __init__(self, realm, hat=0, top=0, bottom=0, weapon=0):
      self.equipment = {
            Item.Hat:     Item.Hat(realm, hat) if hat !=0 else None,
            Item.Top:     Item.Top(realm, top) if top !=0 else None,
            Item.Bottom:  Item.Bottom(realm, bottom) if bottom != 0 else None,
            Item.Weapon:  Item.Weapon(realm, weapon) if weapon != 0 else None,
            Item.Scrap:   Item.Scrap(realm, 0),
            Item.Shard:   Item.Shard(realm, 0),
            Item.Shaving: Item.Shaving(realm, 0)}

      self.itm = Item.Hat(realm, 0)
  
   def remove(self, item):
      itm = self.get(item, remove=True)
      assert itm, 'item.remove: {} not in inventory'.format(item)
      return itm

   def get(self, item, level=None, remove=False):
      if item not in self.equipment:
          return None

      itm = self.equipment[item]

      if remove:
          self.equipment[item] = None

      return itm

   def use_ammunition(self, skill):
      if skill == Skill.Mage:
         return self.equipment[Item.Shard].use()
      elif skill == Skill.Range:
         return self.equipment[Item.Shaving].use()
      elif skill == Skill.Melee:
         return self.equipment[Item.Scrap].use()
      else:
         assert False, 'No ammunition for skill {}'.format(skill)

   def add(self, ammo):
      ammo_type = type(ammo)
      item = self.equipment[ammo_type]
      item.quantity.update(item.quantity.val + ammo.quantity.val)

   @property
   def packet(self):
      packet = {}

      for item_type, item in self.equipment.items():
          name = item_type.__name__
          if item:
              packet[name] = item.packet
          else:
              packet[name] = self.itm.packet
         
      return packet

   @property
   def items(self):
      return [e for e in self.equipment.values() if e is not None]

   @property
   def levels(self):
      return [e.level.val for e in self.items]

   @property
   def defense(self):
      if not (items := self.items):
         return 0
      return sum(e.defense.val for e in items)

   @property
   def offense(self):
      if not (items := self.items):
         return 0
      return sum(e.offense.val for e in items)

   @property
   def level(self):
      if not (levels := self.levels):
         return 0
      return sum(levels)
         
class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self.gold        = Item.Gold(realm)
      self.equipment   = Loadout(realm)
      self.consumables = Pouch(config.N_CONSUMABLES)
      self.loot        = Pouch(config.N_LOOT)

      self.pouches = [self.equipment, self.consumables, self.loot]

   def packet(self):
      data                = {}

      data['gold']        = self.gold.packet
      data['equipment']   = self.equipment.packet
      data['consumables'] = self.consumables.packet
      data['loot']        = self.loot.packet

      return data

   @property
   def items(self):
      return (self.equipment.items + [self.gold] + 
            self.consumables.items + self.loot.items)

   @property
   def dataframeKeys(self):
      return [e.instanceID for e in self.items]

   def remove(self, item, level=None):
      itm = self.get(item, level, remove=True)
      assert itm, 'item.remove: {} not in inventory'.format(item)
      return itm

   def get(self, item, level=None, remove=False):
      for pouch in self.pouches:
          itm = pouch.get(item, level, remove)
          if itm:
             return itm

   def use(self, item):
      assert self.get(type(item))
      item.use(self.entity)

   def receive(self, items):
      if type(items) != list:
         items = [items]

      for item in items:
         #msg = 'Received Drop: Level {} {}'
         #print(msg.format(item.level, item.__class__.__name__))
         if isinstance(item, Item.Gold):
            self.gold.quantity += item.quantity.val
         elif isinstance(item, Item.Ammunition):
            self.equipment.add(item)
         elif self.loot.space:
            self.loot.add(item)
