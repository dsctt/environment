import numpy as np
from typing import Dict, List

from nmmo.core.realm import Realm

class Task():
  def completed(self, realm: Realm) -> bool:
    raise NotImplementedError

  def to_string(self) -> str:
    raise NotImplementedError

###############################################################

class TaskTarget(object):
  def __init__(self, name: str, agents: List[str]) -> None:
    self._name = name
    self._agents = agents

  def agents(self) ->  List[int]:
    return self._agents

  def to_string(self) -> str:
    return self._name

class TargetTask(Task):
  def __init__(self, target: TaskTarget) -> None:
    self._target = target

  def completed(self, realm: Realm) -> bool:
    raise NotImplementedError

###############################################################

class TeamHelper(object):
  def __init__(self, agents: List[int], num_teams: int) -> None:
    assert len(agents) % num_teams == 0
    self._teams = np.array_split(agents, num_teams)
    self._agent_to_team = {a: t for t in self._teams for a in t}


###############################################################

class AND(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    self._tasks = tasks

  def completed(self, realm: Realm) -> bool:
    return all([t.completed(realm) for t in self._tasks])

  def to_string(self) -> str:
    return "(AND " + [t.to_string() for t in self._tasks] + ")"

class OR(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    self._tasks = tasks

  def completed(self, realm: Realm) -> bool:
    return any([t.completed(realm) for t in self._tasks])

  def to_string(self) -> str:
    return "(OR " + [t.to_string() for t in self._tasks] + ")"

class NOT(Task):
  def __init__(self, task: Task) -> None:
    super().__init__()
    self._task = task

  def completed(self, realm: Realm) -> bool:
    return not self._task.completed(realm)

  def to_string(self) -> str:
    return "(NOT " + self._task.to_string() + ")"

###############################################################

class InflictDamage(TargetTask):
  def __init__(self, target: TaskTarget, damage_type: int, quantity: int):
    super().__init__(target)
    self._damage_type = damage_type
    self._quantiy = quantity

  def completed(self, realm: Realm) -> bool:
    # TODO(daveey) damage_type is ignored, needs to be added to entity.history
    return sum([
      realm.players[a].history.damage_inflicted for a in self._target.agents()
    ]) >= self._quantiy

class Defend(TargetTask):
  def __init__(self, target) -> None:
    super().__init__(target)

  def completed(self, realm: Realm) -> bool:
    # TODO(daveey) need a way to specify time horizon
    return realm.tick >= 1024 and all([
      realm.players[a].alive for a in self._target.agents()
    ])

###############################################################

# class TaskParser():
#   def __init__(self) -> None:
#     self.parsers = dict()

#     self.register(InflictDamage)
#     self.register(Defend)

#     self.register(AND)
    
#     self.register(Team)
  
#   def register(self, task_class):
#     self.parsers[task_class.__name__] = task_class

#   def parse(task_string: str):
#     assert task_string.startswith("(") and task_string.endswith(")")
#     parts = task_string[1:-1].split(" ")


# AND(InflictDamage(Team.LEFT, 1, 10), Defend(Team.SELF.Member(0)))
    
# """
#   (AND (InflictDamage Team.LEFT MELEE 5) (Defend Team.SELF.1))
# """