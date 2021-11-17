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
         melee_attack=0, range_attack=0, mage_attack=0,
         melee_defense=0, range_defense=0, mage_defense=0,
         health_restore=0, resource_restore=0, price=0):

      self.config     = realm.config
      self.realm      = realm  

      self.instanceID = Item.INSTANCE_ID
      realm.items[self.instanceID] = self

      self.instance         = Static.Item.ID(realm.dataframe, self.instanceID, Item.INSTANCE_ID)
      self.index            = Static.Item.Index(realm.dataframe, self.instanceID, self.ITEM_ID)
      self.level            = Static.Item.Level(realm.dataframe, self.instanceID, level)
      self.capacity         = Static.Item.Capacity(realm.dataframe, self.instanceID, capacity)
      self.quantity         = Static.Item.Quantity(realm.dataframe, self.instanceID, quantity)
      self.tradable         = Static.Item.Tradable(realm.dataframe, self.instanceID, tradable)
      self.melee_attack     = Static.Item.MeleeAttack(realm.dataframe, self.instanceID, melee_attack)
      self.range_attack     = Static.Item.RangeAttack(realm.dataframe, self.instanceID, range_attack)
      self.mage_attack      = Static.Item.MageAttack(realm.dataframe, self.instanceID, mage_attack)
      self.melee_defense    = Static.Item.MeleeDefense(realm.dataframe, self.instanceID, melee_defense)
      self.range_defense    = Static.Item.RangeDefense(realm.dataframe, self.instanceID, range_defense)
      self.mage_defense     = Static.Item.MageDefense(realm.dataframe, self.instanceID, mage_defense)
      self.health_restore   = Static.Item.HealthRestore(realm.dataframe, self.instanceID, health_restore)
      self.resource_restore = Static.Item.ResourceRestore(realm.dataframe, self.instanceID, resource_restore)
      self.price            = Static.Item.Price(realm.dataframe, self.instanceID, price)
      self.equipped         = Static.Item.Equipped(realm.dataframe, self.instanceID, 0)

      realm.dataframe.init(Static.Item, self.instanceID, None)

      Item.INSTANCE_ID += 1
      if self.ITEM_ID is not None:
         ItemID.register(self.__class__, item_id=self.ITEM_ID)

   @property
   def signature(self):
      return (self.index.val, self.level.val)

   @property
   def packet(self):
      return {'item':             self.__class__.__name__,
              'level':            self.level.val,
              'capacity':         self.capacity.val,
              'quantity':         self.quantity.val,
              'melee_attack':     self.melee_attack.val,
              'range_attack':     self.range_attack.val,
              'mage_attack':      self.mage_attack.val,
              'melee_defense':    self.melee_defense.val,
              'range_defense':    self.range_defense.val,
              'mage_defense':     self.mage_defense.val,
              'health_restore':   self.health_restore.val,
              'resource_restore': self.resource_restore.val,
              'price':            self.price.val}
 
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
      super().__init__(realm, level,
              melee_defense=defense,
              range_defense=defense,
              mage_defense=defense,
              **kwargs)

class Hat(Armor):
   ITEM_ID = 2

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.hat:
          entity.inventory.equipment.hat.use(entity)
      entity.inventory.equipment.hat = self

   def unequip(self, entity):
      entity.inventory.equipment.hat = None

class Top(Armor):
   ITEM_ID = 3

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.top:
          entity.inventory.equipment.top.use(entity)
      entity.inventory.equipment.top = self

   def unequip(self, entity):
      entity.inventory.equipment.top = None

class Bottom(Armor):
   ITEM_ID = 4

   def equip(self, entity):
      if entity.level < self.level.val:
          return
      if entity.inventory.equipment.bottom:
          entity.inventory.equipment.bottom.use(entity)
      entity.inventory.equipment.bottom = self

   def unequip(self, entity):
      entity.inventory.equipment.bottom = None

class Weapon(Equipment):
   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None

class Sword(Weapon):
   ITEM_ID = 5
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, melee_attack=realm.config.EQUIPMENT_OFFENSE(level), **kwargs)

   def equip(self, entity):
      if entity.skills.melee.level.val >= self.level.val:
         super().equip(entity)
 
class Bow(Weapon):
   ITEM_ID = 6
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, range_attack=realm.config.EQUIPMENT_OFFENSE(level), **kwargs)

   def equip(self, entity):
      if entity.skills.range.level.val >= self.level.val:
         super().equip(entity)
 
class Wand(Weapon):
   ITEM_ID = 7
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, mage_attack=realm.config.EQUIPMENT_OFFENSE(level), **kwargs)

   def equip(self, entity):
      if entity.skills.mage.level.val >= self.level.val:
         super().equip(entity)
 
class Tool(Equipment):
   def __init__(self, realm, level, **kwargs):
      defense = realm.config.TOOL_DEFENSE(level)
      super().__init__(realm, level,
              melee_defense=defense,
              range_defense=defense,
              mage_defense=defense,
              **kwargs)

   def equip(self, entity):
      if entity.inventory.equipment.held:
          entity.inventory.equipment.held.use(entity)
      entity.inventory.equipment.held = self

   def unequip(self, entity):
      entity.inventory.equipment.held = None

class Rod(Tool):
    ITEM_ID = 8
    def equip(self, entity):
       if entity.skills.fishing.level >= self.level.val:
          super().equip(entity)

class Gloves(Tool):
    ITEM_ID = 9
    def equip(self, entity):
       if entity.skills.herbalism.level >= self.level.val:
          super().equip(entity)

class Pickaxe(Tool):
    ITEM_ID = 10
    def equip(self, entity):
       if entity.skills.prospecting.level >= self.level.val:
          super().equip(entity)

class Chisel(Tool):
    ITEM_ID = 11
    def equip(self, entity):
       if entity.skills.carving.level >= self.level.val:
          super().equip(entity)

class Arcane(Tool):
    ITEM_ID = 12
    def equip(self, entity):
       if entity.skills.alchemy.level >= self.level.val:
          super().equip(entity)


class Ammunition(Stack):
   def use(self, entity):
      if __debug__:
         err = 'Used ammunition with 0 quantity'
         assert self.quantity.val > 0, err

      self.quantity.decrement()

      if self.quantity.val == 0:
         entity.inventory.remove(self)

      return self.damage
      
class Scrap(Ammunition):
   ITEM_ID = 13
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, melee_attack=realm.config.DAMAGE_AMMUNITION(level), **kwargs)

   @property
   def damage(self):
      return self.melee_attack.val

class Shaving(Ammunition):
   ITEM_ID = 14
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, range_attack=realm.config.DAMAGE_AMMUNITION(level), **kwargs)

   @property
   def damage(self):
      return self.range_attack.val

class Shard(Ammunition):
   ITEM_ID = 15
   def __init__(self, realm, level, **kwargs):
      super().__init__(realm, level, mage_attack=realm.config.DAMAGE_AMMUNITION(level), **kwargs)

   @property
   def damage(self):
      return self.mage_attack.val

class Consumable(Item):
    pass

class Ration(Consumable):
   ITEM_ID = 16
   def __init__(self, realm, level, **kwargs):
      restore = realm.config.RESTORE(level)
      super().__init__(realm, level, resource_restore=restore, **kwargs)

   def use(self, entity):
      entity.resources.food.increment(self.restore.val)
      entity.resources.water.increment(self.restore.val)

class Poultice(Consumable):
   ITEM_ID = 17

   def __init__(self, realm, level, **kwargs):
      restore = realm.config.RESTORE(level)
      super().__init__(realm, level, health_restore=restore, **kwargs)

   def use(self, entity):
      entity.resources.health.increment(self.restore.val)
 
