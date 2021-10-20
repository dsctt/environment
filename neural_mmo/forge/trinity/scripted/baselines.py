from pdb import set_trace as T
import numpy as np

from neural_mmo.forge.trinity.agent import Agent
from neural_mmo.forge.trinity.scripted import behavior, move, attack, utils, io
from neural_mmo.forge.blade.io.stimulus.static import Stimulus
from neural_mmo.forge.blade.io.action import static as Action
from neural_mmo.forge.blade.lib import material
from neural_mmo.forge.blade.lib import enums
from neural_mmo.forge.blade import item

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
        self.inventory  = set()
        self.best_items = {}

        self.gold = io.Observation.attribute(self.ob.agent, Stimulus.Entity.Gold)

        for item_ary in self.ob.items:
           index    = io.Observation.attribute(item_ary, Stimulus.Item.Index)
           level    = io.Observation.attribute(item_ary, Stimulus.Item.Level)
           quantity = io.Observation.attribute(item_ary, Stimulus.Item.Quantity)
           instance = io.Observation.attribute(item_ary, Stimulus.Item.ID)

           if quantity == 0:
              continue

           itm      = item.ItemID.get(int(index))
           self.inventory.add((itm, instance, level, quantity))

           if itm not in self.best_items:
              self.best_items[itm] = (instance, level, quantity)

           _ , best, _ = self.best_items[itm]
           if best < level:
              self.best_items[itm] = (instance, level, quantity)

    def process_market(self):
        self.market          = set()
        self.best_affordable = {}

        for item_ary in self.ob.market:
           index    = io.Observation.attribute(item_ary, Stimulus.Item.Index)
           level    = io.Observation.attribute(item_ary, Stimulus.Item.Level)
           quantity = io.Observation.attribute(item_ary, Stimulus.Item.Quantity)
           price    = io.Observation.attribute(item_ary, Stimulus.Item.Price)
           instance = io.Observation.attribute(item_ary, Stimulus.Item.ID)

           itm      = item.ItemID.get(int(index))

           self.market.add((itm, instance, level, quantity, price))

           #Affordable
           if price > self.gold:
              continue

           if itm not in self.best_affordable:
               self.best_affordable[itm] = (instance, level, quantity, price)

           _, best_level, _, best_price = self.best_affordable[itm]

           #Not lower level
           if level < best_level:
               continue

           #Not same level but more expensive
           if level == best_level and price > best_price:
               continue

           self.best_affordable[itm] = (instance, level, quantity, price)

 
    def sell(self, keep_all: set, keep_best: set):
        for itm, instance, level, quantity in self.inventory:
            if itm in keep_all:
                continue

            if itm == item.Gold:
                continue

            best = level == self.best_items[itm][0]
            if itm in keep_best and best:
                continue

            if quantity == 0:
                continue

            self.actions[Action.Exchange] = {
               Action.ExchangeAction: Action.Sell,
               Action.Item: instance,
               Action.Quantity: quantity,
               Action.Price: level}

            return True

    def buy(self, buy_best: set, buy_upgrade: set):
        purchase = None
        for item, (instance, level, quantity, price) in self.best_affordable.items():
            if item in buy_best:
                purchase = (item, instance, level, quantity, price)
            if item not in buy_upgrade:
                continue
            if item not in self.best_items:
                purchase = (item, instance, level, quantity, price)
            if self.best_items[item][0] < level:
                purchase = (item, instance, level, quantity, price)

        if purchase is None:
            return
 
        item, instance, level, quantity, price = purchase
        self.actions[Action.Exchange] = {
           Action.ExchangeAction: Action.Buy,
           Action.Item: instance,
           Action.Quantity: 1}

        return True
 
    def exchange(self):
        rand = np.random.rand()
        if rand < 0.25:
           self.actions[Action.Exchange] = {
                 Action.ExchangeAction: Action.Buy,
                 Action.Item: item.Food,
                 Action.Level: 1,
                 Action.Quantity: 1,
                 Action.Price: 1}
        elif rand < 0.5:
            self.actions[Action.Exchange] = {
                 Action.ExchangeAction: Action.Sell,
                 Action.Item: item.Food,
                 Action.Level: 1,
                 Action.Quantity: 1,
                 Action.Price: 2}
        elif rand < 0.75:
            self.actions[Action.Inventory] = {
                 Action.InventoryAction: Action.Use,
                 Action.Item: item.Food}
        else:
            self.actions[Action.Inventory] = {
                 Action.InventoryAction: Action.Discard,
                 Action.Item: item.Food}

    def exchange_resources(self, action):
        rand = np.random.rand()
        if rand < 1/3.0:
            itm = item.Scrap
        elif rand < 2/3.0:
            itm = item.Shaving
        else:
            itm = item.Shard

        self.actions[Action.Exchange] = {
              Action.ExchangeAction: action,
              Action.Item: itm,
              Action.Level: 1,
              Action.Quantity: 1,
              Action.Price: 2}

    def exchange_equipment(self, action):
        rand = np.random.rand()
        if rand < 1/3.0:
            itm = item.Hat
        elif rand < 2/3.0:
            itm = item.Top
        else:
            itm = item.Bottom

        self.actions[Action.Exchange] = {
              Action.ExchangeAction: action,
              Action.Level: 1,
              Action.Item: itm,
              Action.Quantity: 1,
              Action.Price: 5}

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

        if self.forage_criterion:
           self.forage()
        elif self.gather(self.resource):
            pass #Successful pathing
        else:
           self.explore()


        item_sold = self.sell(
                keep_all={},
                keep_best={item.Hat, item.Top, item.Bottom, item.Tool})

        if item_sold:
           return self.actions

        item_bought = self.buy(
                buy_best={item.Hat, item.Top, item.Bottom, item.Tool},
                buy_upgrade={})

        return self.actions

class Prospector(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Ore

class Hunter(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Herb

class Fisher(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Fish

class Carver(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Tree

class Alchemist(Gather):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.resource = material.Crystal

class CombatExchange(Combat):
    @property
    def policy(self):
       return self.__class__.__name__

    def __call__(self, obs):
        super().__call__(obs)
        self.process_inventory()
        self.process_market()

        item_sold = self.sell(
                keep_all={item.Ration, item.Potion, self.ammo},
                keep_best={item.Hat, item.Top, item.Bottom, item.Weapon})

        if not item_sold:
           return self.actions

        item_bought = self.buy(
                buy_best={item.Ration, item.Potion, self.ammo},
                buy_upgrade={})

        return self.actions

class Melee(CombatExchange):
    policy = 'Melee'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.style = Action.Melee
        self.ammo  = item.Scrap

class Range(CombatExchange):
    policy = 'Range'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.style = Action.Range
        self.ammo  = item.Shaving

class Mage(CombatExchange):
    policy = 'Mage'
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.style = Action.Mage
        self.ammo  = item.Shard

