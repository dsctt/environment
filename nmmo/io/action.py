# pylint: disable=all
# TODO(kywch): If edits work, I will make it pass pylint
#   also the env in call(env, entity, direction) functions below
#   is actually realm. See realm.step() Should be changed.

from ordered_set import OrderedSet
import numpy as np

from enum import Enum, auto

from nmmo.lib import utils
from nmmo.lib.utils import staticproperty
from nmmo.systems.item import Item, Stack

class NodeType(Enum):
   #Tree edges
   STATIC = auto()    #Traverses all edges without decisions
   SELECTION = auto() #Picks an edge to follow

   #Executable actions
   ACTION    = auto() #No arguments
   CONSTANT  = auto() #Constant argument
   VARIABLE  = auto() #Variable argument

class Node(metaclass=utils.IterableNameComparable):
   @classmethod
   def init(cls, config):
       pass

   @staticproperty
   def edges():
      return []

   #Fill these in
   @staticproperty
   def priority():
      return None

   @staticproperty
   def type():
      return None

   @staticproperty
   def leaf():
      return False

   @classmethod
   def N(cls, config):
      return len(cls.edges)

   def deserialize(realm, entity, index):
      return index

   def args(stim, entity, config):
      return []

class Fixed:
   pass

#ActionRoot
class Action(Node):
   nodeType = NodeType.SELECTION
   hooked   = False

   @classmethod
   def init(cls, config):
      # Sets up serialization domain
      if Action.hooked:
          return

      Action.hooked = True

   #Called upon module import (see bottom of file)
   #Sets up serialization domain
   def hook(config):
      idx = 0
      arguments = []
      for action in Action.edges(config):
         action.init(config)
         for args in action.edges:
            args.init(config)
            if not 'edges' in args.__dict__:
               continue
            for arg in args.edges:
               arguments.append(arg)
               arg.serial = tuple([idx])
               arg.idx = idx
               idx += 1
      Action.arguments = arguments

   @staticproperty
   def n():
      return len(Action.arguments)

   @classmethod
   def edges(cls, config):
      '''List of valid actions'''
      edges = [Move]
      if config.COMBAT_SYSTEM_ENABLED:
          edges.append(Attack)
      if config.ITEM_SYSTEM_ENABLED:
          edges += [Use, Give, Destroy]
      if config.EXCHANGE_SYSTEM_ENABLED:
          edges += [Buy, Sell, GiveGold]
      if config.COMMUNICATION_SYSTEM_ENABLED:
          edges.append(Comm)
      return edges

   def args(stim, entity, config):
      raise NotImplementedError

class Move(Node):
   priority = 60
   nodeType = NodeType.SELECTION
   def call(env, entity, direction):
      assert entity.alive, "Dead entity cannot act"

      r, c  = entity.pos
      ent_id = entity.ent_id
      entity.history.last_pos = (r, c)
      r_delta, c_delta = direction.delta
      rNew, cNew = r+r_delta, c+c_delta

      # One agent per cell
      tile = env.map.tiles[rNew, cNew]

      if entity.status.freeze > 0:
         return

      entity.row.update(rNew)
      entity.col.update(cNew)

      env.map.tiles[r, c].remove_entity(ent_id)
      env.map.tiles[rNew, cNew].add_entity(entity)

      if env.map.tiles[rNew, cNew].lava:
         entity.receive_damage(None, entity.resources.health.val)

   @staticproperty
   def edges():
      return [Direction]

   @staticproperty
   def leaf():
      return True

class Direction(Node):
   argType = Fixed

   @staticproperty
   def edges():
      return [North, South, East, West]

   def args(stim, entity, config):
      return Direction.edges

class North(Node):
   delta = (-1, 0)

class South(Node):
   delta = (1, 0)

class East(Node):
   delta = (0, 1)

class West(Node):
   delta = (0, -1)


class Attack(Node):
   priority = 50
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Style, Target]

   @staticproperty
   def leaf():
      return True

   def inRange(entity, stim, config, N):
      R, C = stim.shape
      R, C = R//2, C//2

      rets = OrderedSet([entity])
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].entities.values():
               rets.add(e)

      rets = list(rets)
      return rets

   # CHECK ME: do we need l1 distance function?
   #   systems/ai/utils.py also has various distance functions
   #   which we may want to clean up
   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(env, entity, style, targ):
      assert entity.alive, "Dead entity cannot act"
      
      config = env.config
      if entity.is_player and not config.COMBAT_SYSTEM_ENABLED:
         return

      # Testing a spawn immunity against old agents to avoid spawn camping
      immunity = config.COMBAT_SPAWN_IMMUNITY
      if entity.is_player and targ.is_player and \
         targ.history.time_alive < immunity < entity.history.time_alive.val:
         return

      #Check if self targeted
      if entity.ent_id == targ.ent_id:
         return

      #ADDED: POPULATION IMMUNITY
      if not config.COMBAT_FRIENDLY_FIRE and entity.is_player and entity.population_id.val == targ.population_id.val:
         return

      #Can't attack out of range
      if utils.linf(entity.pos, targ.pos) > style.attackRange(config):
         return

      #Execute attack
      entity.history.attack = {}
      entity.history.attack['target'] = targ.ent_id
      entity.history.attack['style'] = style.__name__
      targ.attacker = entity
      targ.attacker_id.update(entity.ent_id)

      from nmmo.systems import combat
      dmg = combat.attack(env, entity, targ, style.skill)

      if style.freeze and dmg > 0:
         targ.status.freeze.update(config.COMBAT_FREEZE_TIME)

      return dmg

class Style(Node):
   argType = Fixed
   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   def args(stim, entity, config):
      return Style.edges


class Target(Node):
   argType = None

   @classmethod
   def N(cls, config):
      return config.PLAYER_N_OBS

   def deserialize(realm, entity, index):
      # NOTE: index is the entity id
      # CHECK ME: should index be renamed to ent_id?
      return realm.entity(index)

   def args(stim, entity, config):
      #Should pass max range?
      return Attack.inRange(entity, stim, config, None)

class Melee(Node):
   nodeType = NodeType.ACTION
   freeze=False

   def attackRange(config):
      return config.COMBAT_MELEE_REACH

   def skill(entity):
      return entity.skills.melee

class Range(Node):
   nodeType = NodeType.ACTION
   freeze=False

   def attackRange(config):
      return config.COMBAT_RANGE_REACH

   def skill(entity):
      return entity.skills.range

class Mage(Node):
   nodeType = NodeType.ACTION
   freeze=False

   def attackRange(config):
      return config.COMBAT_MAGE_REACH

   def skill(entity):
      return entity.skills.mage


class InventoryItem(Node):
    argType  = None

    @classmethod
    def N(cls, config):
        return config.INVENTORY_N_OBS

    # TODO(kywch): What does args do?
    def args(stim, entity, config):
        return stim.exchange.items()

    def deserialize(realm, entity, index):
        # NOTE: index is from the inventory, NOT item id
        inventory = Item.Query.owned_by(realm.datastore, entity.id.val)

        if index >= inventory.shape[0]:
            return None

        item_id = inventory[index, Item.State.attr_name_to_col["id"]]
        return realm.items[item_id]

class Use(Node):
    priority = 10

    @staticproperty
    def edges():
        return [InventoryItem]

    def call(env, entity, item):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot use an item"
        assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

        if not env.config.ITEM_SYSTEM_ENABLED:
           return

        if item not in entity.inventory:
            return

        # cannot use listed items or items that have higher level
        if item.listed_price.val > 0 or item.level.val > item._level(entity):
            return

        return item.use(entity)

class Destroy(Node):
    priority = 40

    @staticproperty
    def edges():
        return [InventoryItem]

    def call(env, entity, item):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot destroy an item"
        assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

        if not env.config.ITEM_SYSTEM_ENABLED:
           return

        if item not in entity.inventory:
            return

        if item.equipped.val: # cannot destroy equipped item
            return
        
        # inventory.remove() also unlists the item, if it has been listed
        entity.inventory.remove(item)

        return item.destroy()

class Give(Node):
    priority = 30

    @staticproperty
    def edges():
        return [InventoryItem, Target]

    def call(env, entity, item, target):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot give an item"
        assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

        config = env.config
        if not config.ITEM_SYSTEM_ENABLED:
           return

        if not (target.is_player and target.alive):
           return

        if item not in entity.inventory:
            return

        # cannot give the equipped or listed item
        if item.equipped.val or item.listed_price.val:
            return

        if not (config.ITEM_GIVE_TO_FRIENDLY and
                entity.population_id == target.population_id and        # the same team
                entity.ent_id != target.ent_id and                      # but not self
                utils.linf(entity.pos, target.pos) == 0):               # the same tile
            return

        if not target.inventory.space:
            # receiver inventory is full - see if it has an ammo stack with the same sig
            if isinstance(item, Stack):
               if not target.inventory.has_stack(item.signature):
                  # no ammo stack with the same signature, so cannot give
                  return
            else: # no space, and item is not ammo stack, so cannot give 
               return

        entity.inventory.remove(item)
        return target.inventory.receive(item)


class GiveGold(Node):
    priority = 30

    @staticproperty
    def edges():
        # CHECK ME: for now using Price to indicate the gold amount to give
        return [Target, Price]

    def call(env, entity, target, amount):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot give gold"

        config = env.config
        if not config.EXCHANGE_SYSTEM_ENABLED:
            return

        if not (target.is_player and target.alive):
           return

        if not (config.ITEM_GIVE_TO_FRIENDLY and
                entity.population_id == target.population_id and        # the same team
                entity.ent_id != target.ent_id and                      # but not self
                utils.linf(entity.pos, target.pos) == 0):               # the same tile
            return

        if type(amount) != int:
            amount = amount.val

        if not (amount > 0 and entity.gold.val > 0): # no gold to give
           return

        amount = min(amount, entity.gold.val)

        entity.gold.decrement(amount)
        return target.gold.increment(amount)


class MarketItem(Node):
    argType  = None

    @classmethod
    def N(cls, config):
        return config.MARKET_N_OBS

    # TODO(kywch): What does args do?
    def args(stim, entity, config):
        return stim.exchange.items()

    def deserialize(realm, entity, index):
        # NOTE: index is from the market, NOT item id
        market = Item.Query.for_sale(realm.datastore)

        if index >= market.shape[0]:
            return None
        
        item_id = market[index, Item.State.attr_name_to_col["id"]]
        return realm.items[item_id]

class Buy(Node):
    priority = 20
    argType  = Fixed

    @staticproperty
    def edges():
        return [MarketItem]

    def call(env, entity, item):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot buy an item"
        assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak
        assert item.equipped.val == 0, 'Listed item must not be equipped'

        if not env.config.EXCHANGE_SYSTEM_ENABLED:
            return

        if entity.gold.val < item.listed_price.val: # not enough money
            return
        
        if entity.ent_id == item.owner_id.val: # cannot buy own item
            return

        if not entity.inventory.space:
            # buyer inventory is full - see if it has an ammo stack with the same sig
            if isinstance(item, Stack):
               if not entity.inventory.has_stack(item.signature):
                  # no ammo stack with the same signature, so cannot give
                  return
            else: # no space, and item is not ammo stack, so cannot give 
               return

        # one can try to buy, but the listing might have gone (perhaps bought by other)
        return env.exchange.buy(entity, item)

class Sell(Node):
    priority = 70
    argType  = Fixed

    @staticproperty
    def edges():
        return [InventoryItem, Price]

    def call(env, entity, item, price):
        assert entity.alive, "Dead entity cannot act"
        assert entity.is_player, "Npcs cannot sell an item"
        assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

        if not env.config.EXCHANGE_SYSTEM_ENABLED:
            return

        # TODO: Find a better way to check this
        # Should only occur when item is used on same tick
        # Otherwise should not be possible
        #   >> This should involve env._validate_actions, and perhaps action priotities
        if item not in entity.inventory:
            return

        # cannot sell the equipped or listed item
        if item.equipped.val or item.listed_price.val:
            return

        if type(price) != int:
            price = price.val

        if not (price > 0):
           return

        return env.exchange.sell(entity, item, price, env.tick)

def init_discrete(values):
    classes = []
    for i in values:
        name = f'Discrete_{i}'
        cls  = type(name, (object,), {'val': i})
        classes.append(cls)
    return classes

class Price(Node):
    argType  = Fixed

    @classmethod
    def init(cls, config):
        Price.classes = init_discrete(range(1, 101)) # gold should be > 0 

    @staticproperty
    def edges():
        return Price.classes

    def args(stim, entity, config):
        return Price.edges

class Token(Node):
    argType  = Fixed

    @classmethod
    def init(cls, config):
        Comm.classes = init_discrete(range(config.COMMUNICATION_NUM_TOKENS))

    @staticproperty
    def edges():
        return Comm.classes

    def args(stim, entity, config):
        return Comm.edges

class Comm(Node):
    argType  = Fixed
    priority = 99

    @staticproperty
    def edges():
        return [Token]

    def call(env, entity, token):
        entity.message.update(token.val)

#TODO: Solve AGI
class BecomeSkynet:
   pass
