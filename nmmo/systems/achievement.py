from pdb import set_trace as T


from typing import Dict, List
from nmmo.lib.task import Task

class Achievement:
   def __init__(self, task: Task, reward: float):
      self.completed = False
      self.task = task
      self.reward = reward

   @property
   def name(self):
      return self.task.to_string()

   def update(self, realm, entity):
      if self.completed:
         return 0

      if self.task.completed(realm, entity):
         self.completed = True
         return self.reward

      return 0

class Diary:
   def __init__(self, achievements: List[Achievement]):
      self.achievements = achievements

   @property
   def completed(self):
      return sum(a.completed for a in self.achievements)

   @property
   def cumulative_reward(self, aggregate=True):
      return sum(a.reward * a.completed for a in self.achievements)

   def update(self, realm, entity):
      return {a.name: a.update(realm, entity) for a in self.achievements}

