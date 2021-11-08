from pdb import set_trace as T
import random

from neural_mmo.forge.blade.io.stimulus import Static
from neural_mmo.forge.blade.lib.enums import Tier

class ItemID:
   item_ids = {} 
   id_items = {}

   def register(cls, item_id):
      if __debug__:
         if cls in ItemID.item_ids:
            assert ItemID.item_ids[cls] == item_id, 'Missmatched item_id assignment for class {}'.format(cls)
         if item_id in ItemID.id_items:
            assert ItemID.id_items[item_id] == cls, 'Missmatched class assignment for item_id {}'.format(item_id)

      ItemID.item_ids[cls] = item_id
      ItemID.id_items[item_id] = cls

   def get(cls_or_id):
      if type(cls_or_id) == int:
         return ItemID.id_items[cls_or_id]
      return ItemID.item_ids[cls_or_id]

class Item:
   ITEM_ID = None
   INSTANCE_ID = 0
   def __init__(self, realm, level,
         capacity=0, quantity=1, tradable=True,
         offense=0, defense=0, minDmg=0, maxDmg=0,
         restore=0, price=0):

      self.config     = realm.config
      self.realm      = realm  

      self.instanceID = Item.INSTANCE_ID
      realm.items[self.instanceID] = self

      self.instance = Static.Item.ID(realm.dataframe, self.instanceID, Item.INSTANCE_ID)
      self.index    = Static.Item.Index(realm.dataframe, self.instanceID, self.ITEM_ID)
      self.level    = Static.Item.Level(realm.dataframe, self.instanceID, level)
      self.capacity = Static.Item.Capacity(realm.dataframe, self.instanceID, capacity)
      self.quantity = Static.Item.Quantity(realm.dataframe, self.instanceID, quantity)
      self.tradable = Static.Item.Tradable(realm.dataframe, self.instanceID, tradable)
      self.offense  = Static.Item.Offense(realm.dataframe, self.instanceID, offense)
      self.defense  = Static.Item.Defense(realm.dataframe, self.instanceID, defense)
      self.minDmg   = Static.Item.MinDmg(realm.dataframe, self.instanceID, minDmg)
      self.maxDmg   = Static.Item.MaxDmg(realm.dataframe, self.instanceID, maxDmg)
      self.restore  = Static.Item.Restore(realm.dataframe, self.instanceID, restore)
      self.price    = Static.Item.Price(realm.dataframe, self.instanceID, price)
      self.equipped = Static.Item.Equipped(realm.dataframe, self.instanceID, 0)

      realm.dataframe.init(Static.Item, self.instanceID, None)

      Item.INSTANCE_ID += 1
      if self.ITEM_ID is not None:
         ItemID.register(self.__class__, item_id=self.ITEM_ID)

   @property
   def signature(self):
      return (self.index.val, self.level.val)

   @property
   def packet(self):
      return {'item':     self.__class__.__name__,
              'level':    self.level.val,
              'capacity': self.capacity.val,
              'quantity': self.quantity.val,
              'offense':  self.offense.val,
              'defense':  self.defense.val,
              'minDmg':   self.minDmg.val,
              'maxDmg':   self.maxDmg.val,
              'restore':  self.restore.val,
              'price':    self.price.val}
 
   def use(self, entity):
      return

class Stack(Item):
   def __init__(self, realm, level, quantity=0, **kwargs):
       super().__init__(realm, level, quantity=quantity, **kwargs)

   def use(self, entity):
      assert self.quantity > 0

      self.quantity -= 1
      if self.quantity > 0:
         return

      entity.inventory.remove(self)

class Gold(Item):
   ITEM_ID = 1
   def __init__(self, realm, **kwargs):
      super().__init__(realm, level=0, tradable=False, **kwargs)

class Equipment(Item):
   @property
   def packet(self):
     packet = {'color': self.color.packet()}
     return {**packet, **super().packet}

   @property
   def color(self):
     if self.level == 0:
        return Tier.BLACK
     if self.level < 10:
        return Tier.WOOD
     elif self.level < 20:
        return Tier.BRONZE
     elif self.level < 40:
        return Tier.SILVER
     elif self.level < 60:
        return Tier.GOLD
     elif self.level < 80:
        return Tier.PLATINUM
     else:
        return Tier.DIAMOND

   def use(self, entity):
      if self.equipped.val:
         self.equipped.update(0)
         self.unequip(entity)
      else:
         self.equipped.update(1)
         self.equip(entity)

class Armor(Equipment):
   def __init__(self, realm, level, **kwargs):
      defense = realm.config.EQUIPMENT_DEFENSE(level)
      super().__init__(realm, level, defense=defense, **kwargs)


class Hat(Armor):
   ITEM_ID = 2

   def equip(self, entity):
      if entity.inventory.equipment.hat:
          entity.inventory.equipment.hat.use(entity)
      entity.inventory.equipment.hat = self

   def unequip(self, entity):
      entity.inventory.equipment.hat = None


class Top(Armor):
   ITEM_ID = 3

   def equip(self, entity):
      if entity.inventory.equipment.top:
          entity.inventory.equipment.top.use(entity)
      entity.inventory.equipment.top = self

   def unequip(self, entity):
      entity.inventory.equipment.top = None


class Bottom(Armor):
   ITEM_ID = 4

   def equip(self, entity):
      if entity.inventory.equipment.bottom:
          entity.inventory.equipment.bottom.use(entity)
      entity.inventory.equipment.bottom = self

   def unequip(self, entity):
      entity.inventory.equipment.bottom = None

class Weapon(Equipment):
   ITEM_ID = 5

   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None

class Tool(Armor):
   ITEM_ID = 6

   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None


class Ammunition(Stack):
   def __init__(self, realm, level, **kwargs):
      minDmg, maxDmg = realm.config.DAMAGE_AMMUNITION(level)
      super().__init__(realm, level, minDmg=minDmg, maxDmg=maxDmg, **kwargs)

   def damage(self):
      return random.randint(self.minDmg.val, self.maxDmg.val)

   def use(self):
      if self.quantity.val == 0:
          return 0
      self.quantity.decrement()
      return self.damage()
      
  
class Scrap(Ammunition):
   ITEM_ID = 7

class Shaving(Ammunition):
   ITEM_ID = 8

class Shard(Ammunition):
   ITEM_ID = 9

class Consumable(Item):
   def __init__(self, realm, level, **kwargs):
      restore = realm.config.RESTORE(level)
      super().__init__(realm, level, restore=restore, **kwargs)

class Ration(Consumable):
   ITEM_ID = 10
   def use(self, entity):
      entity.resources.food.increment(self.restore.val)
      entity.resources.water.increment(self.restore.val)

class Potion(Consumable):
   ITEM_ID = 11
   def use(self, entity):
      entity.resources.health.increment(self.restore.val)
 
