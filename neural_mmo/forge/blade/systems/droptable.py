import numpy as np

class Range():
   def __init__(self, mmin, mmax):
      self.mmin = mmin
      self.mmax = mmax

   @property
   def value(self):
      return np.random.randint(self.mmin, self.mmax+1)

class Drop:
   def __init__(self, item, amount, prob=1.0):
      self.item = item
      self.amount = amount
      self.prob = prob

   def roll(self):
      if np.random.rand() <= self.prob:
         if type(self.amount) == int:
            return (self.item(), self.amount)
         return (self.item, self.amount.value)

class DropTable:
   def __init__(self):
      self.drops = []

   def add(self, item, quant, prob=1.0):
      self.drops += [Drop(item, quant, prob)]

   def roll(config, self):
      ret = []
      for e in self.drops:
         drop = e.roll()
         if drop is not None:
            ret += [drop]
      return ret

class Empty(DropTable):
   def roll(self, realm, level):
      return []

class Ammunition(DropTable):
   def __init__(self, item):
      self.item = item
      
   def roll(self, realm, level):
      return self.item(realm, level, quantity=1)

class Consumable(DropTable):
   def __init__(self, item):
      self.item = item
      
   def roll(self, realm, level):
      return self.item(realm, level)

