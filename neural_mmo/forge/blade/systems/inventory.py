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
         if itm is item:
            if remove:
                self.items.remove(itm)
            return itm

         if type(itm) != item:
            continue

         if level is not None and level != itm.level.val:
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


class Equipment:
   def remove(self, item):
      itm = self.get(item, remove=True)
      assert itm, 'item.remove: {} not in inventory'.format(item)
      return itm

   def get(self, item, level=None, remove=False):
      #TODO: Figure out how to replace get/setattr
      if inspect.isclass(item):
         key = item
      else:
         key = type(item)

      if issubclass(key, Item.Weapon) or issubclass(key, Item.Tool):
         key = 'weapon'
      elif issubclass(key, Item.Tool):
         key = 'weapon'
 

      key = key.__lower__

      mapping = self.mapping
      if key not in mapping:
          return None

      itm = getattr(self, key)

      if itm is None:
         return None

      if remove:
         self.force_remove(key)

      return itm

   @property
   def items(self):
      return [e for e in self.mapping.values() if e is not None]

   @property
   def packet(self):
      packet = {}

      for item_type, item in self.mapping.items():
         name = item_type.__name__.lower()

         if item is not None:
            val = item.packet
         else:
            val = self.itm.packet

         packet[name] = val
        
      return packet


class Loadout(Equipment):
   def __init__(self, realm, hat=0, top=0, bottom=0, weapon=0):
      self.hat    = Item.Hat(realm, hat) if hat != 0 else None
      self.top    = Item.Top(realm, top) if top != 0 else None
      self.bottom = Item.Bottom(realm, bottom) if bottom != 0 else None
      self.weapon = Item.Weapon(realm, weapon) if weapon != 0 else None

      self.itm = Item.Hat(realm, 0)

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
 
   @property
   def items(self):
      itms = [self.hat, self.top, self.bottom, self.weapon]
      return [e for e in itms if e is not None]

   @property
   def _mapping(self):
      '''Get only, cannot use to set'''
      return {
            Item.Hat: self.hat,
            Item.Top: self.top,
            Item.Bottom: self.bottom,
            Item.Weapon: self.weapon}

   def force_remove(self, item):
      self.mapping[item] = None


class Ammunition(Equipment):
   def __init__(self, realm, hat=0, top=0, bottom=0, weapon=0):
      self.scrap   = Item.Scrap(realm, 1)
      self.shard   = Item.Shard(realm, 1)
      self.shaving = Item.Shaving(realm, 1)

   @property
   def mapping(self):
      return {
            Item.Scrap:   self.scrap,
            Item.Shard:   self.shard,
            Item.Shaving: self.shaving}

   @property
   def skill_mapping(self):
      return {
            Skill.Melee:  self.scrap,
            Skill.Mage:  self.shard,
            Skill.Range: self.shaving}

   def add(self, ammo):
      ammo_type = type(ammo)
      item = self.mapping[ammo_type]
      item.quantity.update(item.quantity.val + ammo.quantity.val)

   def use(self, skill):
      return self.skill_mapping[skill].use()

   def force_remove(self, item):
      item = self.mapping[item]
      quantity = item.quantity.val
      item.quantity.update(0)

class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self.gold        = Item.Gold(realm)
      self.equipment   = Loadout(realm)
      self.ammunition  = Ammunition(realm)
      self.consumables = Pouch(config.N_CONSUMABLES)
      self.loot        = Pouch(config.N_LOOT)

      self.pouches = [self.equipment, self.ammunition, self.consumables, self.loot]

   def packet(self):
      data = {}

      data['equipment']   = self.equipment.packet
      data['ammunition']  = self.ammunition.packet
      data['gold']        = self.gold.packet
      data['consumables'] = self.consumables.packet
      data['loot']        = self.loot.packet

      return data

   @property
   def items(self):
      return ([self.gold] + self.equipment.items + self.ammunition.items +
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
            self.ammunition.add(item)
         elif self.loot.space:
            self.loot.add(item)

class Equipment:
   def __init__(self, realm):
      self.hat         = None
      self.top         = None
      self.bottom      = None

      self.held        = None
      self.ammunition  = None

      #Placeholder item for render
      self.itm         = Item.Hat(realm, 0)

   @property
   def offense(self): 
      items = [e.offense.val for e in self]
      if not items:
         return 0
      return sum(items)

   @property
   def defense(self): 
      items = [e.defense.val for e in self]
      if not items:
         return 0
      return sum(items)
      
   def __iter__(self):
      for item in [self.hat, self.top, self.bottom, self.held, self.ammunition]:
         if item is not None:
            yield item

   def placeholderize(self, item):
      return item.packet if item is not None else self.itm.packet

   @property
   def packet(self):
      return {
            'hat':        self.placeholderize(self.hat),
            'top':        self.placeholderize(self.top),
            'bottom':     self.placeholderize(self.bottom),
            'held':       self.placeholderize(self.held),
            'ammunition': self.placeholderize(self.ammunition)}
   

class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self._items      = set()
      self.capacity    = config.N_ITEMS

      self.gold        = Item.Gold(realm)
      self.equipment   = Equipment(realm)

   @property
   def space(self):
      return self.capacity - len(self._items)

   @property
   def dataframeKeys(self):
      return [e.instanceID for e in self._items]

   def contains(self, item):
      if item in self._items:
         return True
      return False

   def packet(self):
      return {
            'items':     [e.packet for e in self._items],
            'equipment': self.equipment.packet}

   def __iter__(self):
      for item in self._items:
         yield item

   def receive(self, item):
      if not self.space:
         return
      #space = self.space
      #err = 'Out of space for {}'
      #assert space, err.format(item) 
      self._items.add(item)

   def remove(self, item, level=None):
      err = 'No item {} to remove'
      assert item in self._items, err.format(item)
      self._items.remove(item)
      return item


