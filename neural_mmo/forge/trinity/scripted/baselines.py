from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.trinity.agent import Agent
from neural_mmo.forge.trinity.scripted import behavior, move, attack, utils, io
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.lib import material
from neural_mmo.forge.blade.lib import enums
from neural_mmo.forge.blade import item

from collections import defaultdict

class Item:
    def __init__(self, item_ary): 
        index    = io.Observation.attribute(item_ary, Stimulus.Item.Index)
        self.cls = item.ItemID.get(int(index))

        self.level    = io.Observation.attribute(item_ary, Stimulus.Item.Level)
        self.quantity = io.Observation.attribute(item_ary, Stimulus.Item.Quantity)
        self.price    = io.Observation.attribute(item_ary, Stimulus.Item.Price)
        self.instance = io.Observation.attribute(item_ary, Stimulus.Item.ID)
        self.equipped = io.Observation.attribute(item_ary, Stimulus.Item.Equipped)


class Scripted(Agent):
    '''Template class for scripted models.

    You may either subclass directly or mirror the __call__ function'''
    scripted = True
    color    = enums.Neon.SKY
    def __init__(self, config, idx):
        '''
        Args:
           config : A forge.blade.core.Config object or subclass object
        ''' 
        super().__init__(config, idx)
        self.food_max  = 0
        self.water_max = 0

        self.spawnR    = None
        self.spawnC    = None

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water'''
        min_level = 7
        return self.food <= min_level or self.water <= min_level

    def forage(self):
        '''Min/max food and water using Dijkstra's algorithm'''
        move.forageDijkstra(self.config, self.ob, self.actions, self.food_max, self.water_max)

    def gather(self, resource):
        '''BFS search for a particular resource'''
        return move.gatherBFS(self.config, self.ob, self.actions, resource)

    def explore(self):
        '''Route away from spawn'''
        move.explore(self.config, self.ob, self.actions, self.spawnR, self.spawnC)

    @property
    def downtime(self):
        '''Return true if agent is not occupied with a high-priority action'''
        return not self.forage_criterion and self.attacker is None

    def evade(self):
        '''Target and path away from an attacker'''
        move.evade(self.config, self.ob, self.actions, self.attacker)
        self.target     = self.attacker
        self.targetID   = self.attackerID
        self.targetDist = self.attackerDist

    def attack(self):
        '''Attack the current target'''
        if self.target is not None:
           assert self.targetID is not None
           attack.target(self.config, self.actions, self.style, self.targetID)

    def select_combat_style(self):
       '''Select a combat style based on distance from the current target'''
       if self.target is None:
          return

       if self.targetDist <= self.config.COMBAT_MELEE_REACH:
          self.style = Action.Melee
       elif self.targetDist <= self.config.COMBAT_RANGE_REACH:
          self.style = Action.Range
       else:
          self.style = Action.Mage

    def target_weak(self):
        '''Target the nearest agent if it is weak'''
        if self.closest is None:
            return False

        selfLevel  = io.Observation.attribute(self.ob.agent, Stimulus.Entity.Level)
        targLevel  = io.Observation.attribute(self.closest, Stimulus.Entity.Level)
        population = io.Observation.attribute(self.closest, Stimulus.Entity.Population)
        
        if population == -1 or targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
           self.target     = self.closest
           self.targetID   = self.closestID
           self.targetDist = self.closestDist

    def scan_agents(self):
        '''Scan the nearby area for agents'''
        self.closest, self.closestDist   = attack.closestTarget(self.config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(self.config, self.ob)

        self.closestID = None
        if self.closest is not None:
           self.closestID = io.Observation.attribute(self.closest, Stimulus.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
           self.attackerID = io.Observation.attribute(self.attacker, Stimulus.Entity.ID)

        self.target     = None
        self.targetID   = None
        self.targetDist = None

    def adaptive_control_and_targeting(self, explore=True):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()

        if self.attacker is not None:
           self.evade()
           return

        if self.forage_criterion or not explore:
           self.forage()
        else:
           self.explore()

        self.target_weak()

    def process_inventory(self):
        self.inventory   = set()
        self.best_items  = {}
        self.item_counts = defaultdict(int)

        self.gold = io.Observation.attribute(self.ob.agent, Stimulus.Entity.Gold)

        for item_ary in self.ob.items:
           itm = Item(item_ary)
           cls = itm.cls

           if itm.quantity == 0:
              continue

           self.item_counts[cls] += itm.quantity
           self.inventory.add(itm)

           if cls not in self.best_items:
              self.best_items[cls] = itm

           best_itm = self.best_items[cls]

           if itm.level > best_itm.level:
              self.best_items[cls] = itm

           if __debug__:
              err = 'Key {} must be an Item object'.format(cls)
              assert isinstance(self.best_items[cls], Item), err

    def upgrade_heuristic(self, current_level, upgrade_level, price):
        return (upgrade_level - current_level) / price

    def process_market(self):
        self.market         = set()
        self.best_heuristic = {}

        for item_ary in self.ob.market:
           itm = Item(item_ary)
           cls = itm.cls

           self.market.add(itm)

           #Prune Unaffordable
           if itm.price > self.gold:
              continue

           #Current best item level
           current_level = 0
           if cls in self.best_items:
               current_level = self.best_items[cls].level

           itm.heuristic = self.upgrade_heuristic(current_level, itm.level, itm.price)

           #Always count first item
           if cls not in self.best_heuristic:
               self.best_heuristic[cls] = itm
               continue

           #Better heuristic value
           if itm.heuristic > self.best_heuristic[cls].heuristic:
               self.best_heuristic[cls] = itm

    def equip(self, items: set):
        for cls, itm in self.best_items.items():
            if cls not in items:
               continue

            if itm.equipped:
               continue

            self.actions[Action.Inventory] = {
               Action.InventoryAction: Action.Use,
               Action.Item: itm.instance}
           
            return True
 

    def sell(self, keep_k: dict, keep_best: set):
        for itm in self.inventory:
            cls = itm.cls

            if cls in keep_k:
                owned = self.item_counts[cls]
                k     = keep_k[cls]
                if owned <= k:
                    continue
 
            if cls == item.Gold:
                continue

            best_itm = self.best_items[cls]

            if cls in keep_best and itm.instance == best_itm.instance:
                continue

            if itm.quantity == 0:
                continue

            self.actions[Action.Exchange] = {
               Action.ExchangeAction: Action.Sell,
               Action.Item: itm.instance,
               Action.Quantity: itm.quantity,
               Action.Price: itm.level}

            return True

    def buy(self, buy_k: dict, buy_upgrade: set):
        purchase = None
        for cls, itm in self.best_heuristic.items():
            #Buy top k
            if cls in buy_k:
                owned = self.item_counts[cls]
                k     = buy_k[cls]
                if owned < k:
                   purchase = itm

            #Check if item desired
            if cls not in buy_upgrade:
                continue

            #Check is is an upgrade
            if itm.heuristic <= 0:
                continue

            #Buy best heuristic upgrade
            purchase = itm

        if purchase is None:
            return
 
        self.actions[Action.Exchange] = {
           Action.ExchangeAction: Action.Buy,
           Action.Item: purchase.instance,
           Action.Quantity: 1}

        return True
 
    def __call__(self, obs):
        '''Process observations and return actions

        Args:
           obs: An observation object from the environment. Unpack with io.Observation
        '''
        self.actions = {}

        self.ob = io.Observation(self.config, obs)
        agent   = self.ob.agent

        self.food   = io.Observation.attribute(agent, Stimulus.Entity.Food)
        self.water  = io.Observation.attribute(agent, Stimulus.Entity.Water)

        if self.food > self.food_max:
           self.food_max = self.food
        if self.water > self.water_max:
           self.water_max = self.water

        if self.spawnR is None:
            self.spawnR = io.Observation.attribute(agent, Stimulus.Entity.R)
        if self.spawnC is None:
            self.spawnC = io.Observation.attribute(agent, Stimulus.Entity.C)

class Random(Scripted):
    policy = 'Random'
    '''Moves randomly'''
    def __call__(self, obs):
        super().__call__(obs)

        move.random(self.config, self.ob, self.actions)
        return self.actions

class Meander(Scripted):
    policy = 'Meander'
    '''Moves randomly on safe terrain'''
    def __call__(self, obs):
        super().__call__(obs)

        move.meander(self.config, self.ob, self.actions)
        return self.actions

class ForageNoExplore(Scripted):
    '''Forages using Dijkstra's algorithm'''
    policy = 'ForageNE'
    def __call__(self, obs):
        super().__call__(obs)

        self.forage()

        return self.actions

class Forage(Scripted):
    '''Forages using Dijkstra's algorithm and actively explores'''
    policy = 'Forage'
    def __call__(self, obs):
        super().__call__(obs)

        if self.forage_criterion:
           self.forage()
        else:
           self.explore()

        return self.actions

class CombatNoExplore(Scripted):
    '''Forages using Dijkstra's algorithm and fights nearby agents'''
    policy = 'CombatNE'
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting(explore=False)

        self.style = Action.Range
        self.attack()

        return self.actions
 
class Combat(Scripted):
    '''Forages, fights, and explores'''
    policy = 'Combat'
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()
        self.attack()

        return self.actions

class CombatTribrid(Scripted):
    policy = 'CombatTri'
    '''Forages, fights, and explores.

    Uses a slightly more sophisticated attack routine'''
    def __call__(self, obs):
        super().__call__(obs)

        self.adaptive_control_and_targeting()

        self.select_combat_style()
        self.attack()

        return self.actions

class Gather(Scripted):
    '''Forages, fights, and explores'''
    @property
    def policy(self):
       return self.__class__.__name__

    def __call__(self, obs):
        super().__call__(obs)
        self.process_inventory()
        self.process_market()

        self.equip(items={item.Hat, item.Top, item.Bottom, self.tool})

        if self.forage_criterion:
           self.forage()
        elif self.gather(self.resource):
            pass #Successful pathing
        else:
           self.explore()


        item_sold = self.sell(
                keep_k={item.Ration: 2, item.Poultice: 2},
                keep_best={item.Hat, item.Top, item.Bottom, self.tool})

        if item_sold:
           return self.actions

        item_bought = self.buy(
                buy_k={item.Ration: 2, item.Poultice: 2},
                buy_upgrade={item.Hat, item.Top, item.Bottom, self.tool})

        return self.actions

class Fisher(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Fish
        self.tool     = item.Rod

class Herbalist(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Herb
        self.tool     = item.Gloves

class Prospector(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Ore
        self.tool     = item.Pickaxe

class Carver(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Tree
        self.tool     = item.Chisel

class Alchemist(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Crystal
        self.tool     = item.Arcane

class CombatExchange(Combat):
    @property
    def policy(self):
       return self.__class__.__name__

    def __call__(self, obs):
        super().__call__(obs)
        self.process_inventory()
        self.process_market()

        self.equip(items={item.Hat, item.Top, item.Bottom, self.weapon})

        item_sold = self.sell(
                keep_k={item.Ration: 2, item.Poultice: 2, self.ammo: 2},
                keep_best={item.Hat, item.Top, item.Bottom, self.weapon})

        if item_sold:
           return self.actions

        item_bought = self.buy(
                buy_k={item.Ration: 2, item.Poultice: 2, self.ammo: 2},
                buy_upgrade={item.Hat, item.Top, item.Bottom, self.weapon})

        return self.actions

class Melee(CombatExchange):
    policy = 'Melee'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.weapon = item.Sword
        self.style  = Action.Melee
        self.ammo   = item.Scrap

class Range(CombatExchange):
    policy = 'Range'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.weapon = item.Bow
        self.style  = Action.Range
        self.ammo   = item.Shaving

class Mage(CombatExchange):
    policy = 'Mage'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.weapon = item.Wand
        self.style  = Action.Mage
        self.ammo   = item.Shard


