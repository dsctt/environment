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

  def member(self, member):
    assert member < len(self._agents)
    return TaskTarget(f"{self.to_string()}.{member}", [self._agents[member]])

class TargetTask(Task):
  def __init__(self, target: TaskTarget) -> None:
    self._target = target

  def completed(self, realm: Realm) -> bool:
    raise NotImplementedError

###############################################################

class TeamHelper(object):
  def __init__(self, agents: List[int], num_teams: int) -> None:
    assert len(agents) % num_teams == 0
    team_size = len(agents) // num_teams
    self._teams = [
      list(agents[i * team_size : (i+1)*team_size]) 
      for i in range(num_teams)
    ]
    self._agent_to_team = {a: tid for tid, t in enumerate(self._teams) for a in t}

  def own_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Self", self._teams[self._agent_to_team[agent_id]])

  def left_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Left", self._teams[
      (self._agent_to_team[agent_id] -1) % len(self._teams)
    ])

  def right_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Right", self._teams[
      (self._agent_to_team[agent_id] + 1) % len(self._teams)
    ])

  def all(self) -> TaskTarget:
    return TaskTarget("All", list(self._agent_to_team.keys()))

###############################################################

class AND(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    assert len(tasks)
    self._tasks = tasks

  def completed(self, realm: Realm) -> bool:
    return all([t.completed(realm) for t in self._tasks])

  def to_string(self) -> str:
    return "(AND " + " ".join([t.to_string() for t in self._tasks]) + ")"

class OR(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    assert len(tasks)
    self._tasks = tasks

  def completed(self, realm: Realm) -> bool:
    return any([t.completed(realm) for t in self._tasks])

  def to_string(self) -> str:
    return "(OR " + " ".join([t.to_string() for t in self._tasks]) + ")"

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

