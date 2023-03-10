# pylint: disable=all
# This is a prototype. If this direction is correct,
# it will be moved to proper places.

import unittest

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List

import math
import numpy as np
import numpy_indexed as npi

from pettingzoo.utils.env import AgentID

from testhelpers import ScriptedAgentTestConfig

import nmmo
from nmmo.core.config import Config
from nmmo.core.realm import Realm
from nmmo.core.observation import Observation
from nmmo.lib.datastore.datastore import Datastore

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
from nmmo.systems import item as Item
from nmmo.io import action as Action


RANDOM_SEED = 385

@dataclass
class GameState:
  tick: int
  config: Config
  datastore: Datastore # Tile, Entity, Item, Event
  env_obs: Dict[int, Observation]
  ent2pop: Dict[int, int] # key: ent_id, val: pop_id

  # - add extra info that is not in the datastore (e.g., spawn pos)
  # - would IS_WITHIN, TICK, COUNT_DOWN be good here?

  def attr2col(self, state, attr):
    assert state in [TileState, EntityState, ItemState], "Wrong state provided"
    return state.State.attr_name_to_col[attr]

  def get_data(self, state):
    assert state in [EntityState, ItemState], "Wrong state provided"
    return state.Query.table(self.datastore)

  def parse_row(self, state, id: int):
    assert state in [EntityState, ItemState], "Wrong state provided"
    row = state.Query.by_id(self.datastore, id)
    if len(row):
      return state.parse_array(row)

    return None

  def flt_group_by(self, flt_data, grpby_col, sum_col=0):
    # if sum_col = 0, this fn acts as COUNT, otherwise SUM
    g = npi.group_by(flt_data[:,grpby_col])
    result = {}
    for k, v in zip(*g(flt_data[:,sum_col])):
      if sum_col:
        result[k] = sum(v)
      else:
        result[k] = len(v)
    return result


class GameStateGenerator:
  def __init__(self, realm: Realm, config: Config):
    self.config = deepcopy(config)
    self.ent2pop = self._map_ent_team(realm)

  def _map_ent_team(self, realm: Realm): 
    ent2team: Dict[int, int] = {} # key: ent_id, val: pop_id
    for ent_id, ent in realm.players.items():
      ent2team[ent_id] = ent.population
    return ent2team

  def generate(self, realm: Realm, env_obs: Dict[int, Observation]) -> GameState:
    return GameState(
      tick = realm.tick,
      config = deepcopy(self.config),
      datastore = deepcopy(realm.datastore),
      env_obs = env_obs,
      ent2pop = self.ent2pop)

  # TODO(kywch)
  # most entity/item info can be retrieved from the datastore, but some won't.
  # in that case, we need a simple dataclass to pass remaining info
  

class Task:
  '''Basic reward block
  Pass in an instance of Task to the Env to define the rewards of a environment.
  Each Task is assumed to be across entity
  '''
  def __init__(self, reward=1, max_fulfill=math.inf):
    self._reward = reward
    self._max_fulfill = max_fulfill
    self._fulfill_cnt: Dict[int, int] = {} # key: ent_id

    self._gs: GameState = None

    # key: ent_id or pop_id, value: intermediate result
    self._step_output = {}

  def step(self, gs: GameState):
    '''Compute the intermediate, aggregate variable for task evaluation
        for ALL alive players and save to self._step_result'''
    self._gs = gs
    self._step_output = {}

  def evaluate(self, ent_id: int) -> bool:
    '''Evaluate the task for a single agent by comparing
       the agent's data in _step_result (and agent's own)'''
    raise NotImplementedError

  def reward(self, ent_id: int, update_cnt=True) -> float:
    if self.evaluate(ent_id):
      if update_cnt:
        if ent_id in self._fulfill_cnt:
          self._fulfill_cnt[ent_id] += 1
        else:
          self._fulfill_cnt[ent_id] = 1

      # not giving reward if max_fulfill is reached
      if self._max_fulfill < self._fulfill_cnt[ent_id]:
        return 0

      return self._reward

    # not met the condition, so no reward for this tick
    return 0

  def __str__(self):
    return self.__class__.__name__


# CHECK ME: maybe this should be the default task?
class LiveLong(Task):
  # uses the default __init__, step, reward
  def evaluate(self, ent_id: int):
    row = self._gs.parse_row(EntityState, ent_id)
    if row:
      return row.health > 0

    return False


class HoardGold(Task):
  def __init__(self, min_amount: int):
    super().__init__()
    self.min_amount = min_amount

  def evaluate(self, ent_id: int):
    row = self._gs.parse_row(EntityState, ent_id)
    if row:
      return row.gold >= self.min_amount

    return False


# each agent is rewarded by the number of all alive teammates
class TeamSizeGE(Task): # greater than or equal to
  def __init__(self, min_size: int):
    super().__init__()
    self.min_size = min_size
  
  def step(self, gs: GameState):
    super().step(gs)
    data = gs.get_data(EntityState) # 2d numpy data of all the item instances
    flt_idx = data[:,gs.attr2col(EntityState, 'health')] > 0

    # for each team, count the number of alive agents
    self._step_output = \
      gs.flt_group_by(data[flt_idx], gs.attr2col(EntityState, 'population_id'))

  def evaluate(self, ent_id: int):
    pop_id = self._gs.ent2pop[ent_id]
    if pop_id in self._step_output:
      return self._step_output[pop_id] >= self.min_size


class TeamHoardGold(Task):
  def __init__(self, min_amount: int):
    super().__init__()
    self.min_amount = min_amount

  def step(self, gs: GameState):
    super().step(gs)
    data = gs.get_data(EntityState) # 2d numpy data of all the item instances
    flt_idx = data[:,gs.attr2col(EntityState, 'health')] > 0 # alive agents

    # for each team, sum the gold from all members
    self._step_output = \
      gs.flt_group_by(data[flt_idx],
                      grpby_col = gs.attr2col(EntityState, 'population_id'),
                      sum_col = gs.attr2col(EntityState, 'gold') )

  def evaluate(self, ent_id: int):
    pop_id = self._gs.ent2pop[ent_id]
    if pop_id in self._step_output:
      return self._step_output[pop_id] >= self.min_amount

    return False

class OwnItem(Task):
  '''Own an item of a certain type and level (equal or higher)'''
  def __init__(self, item: Item.Item, min_level: int=0, quantity: int=1):
    super().__init__()
    self.item_type = item.ITEM_TYPE_ID
    self.min_level = min_level
    self.quantity = quantity

  def step(self, gs: GameState):
    super().step(gs)
    data = gs.get_data(ItemState) # 2d numpy data of all the item instances
    flt_idx = (data[:,gs.attr2col(ItemState, 'type_id')] == self.item_type) & \
              (data[:,gs.attr2col(ItemState, 'level')] >= self.min_level)
    
    # if an agent owns the item, then self._step_output[ent_id] > 0
    # if not, ent_id not in self._step_output
    self._step_output = \
      gs.flt_group_by(data[flt_idx], gs.attr2col(ItemState, 'owner_id'))

  def evaluate(self, ent_id: int):
    return ent_id in self._step_output

class EquipItem(Task):
  '''Equip an item of a certain type and level (equal or higher)'''
  def __init__(self, item: Item.Equipment, min_level: int=0):
    super().__init__()
    self.item_type = item.ITEM_TYPE_ID
    self.min_level = min_level

  def step(self, gs: GameState):
    super().step(gs)
    data = gs.get_data(ItemState) # 2d numpy data of all the item instances
    flt_idx = (data[:,gs.attr2col(ItemState, 'type_id')] == self.item_type) & \
              (data[:,gs.attr2col(ItemState, 'level')] >= self.min_level) & \
              (data[:,gs.attr2col(ItemState, 'equipped')] > 0)
    
    # if an agent equips the item, then self._step_output[ent_id] = 1
    # if not, ent_id not in self._step_output
    self._step_output = \
      gs.flt_group_by(data[flt_idx], gs.attr2col(ItemState, 'owner_id'))

  def evaluate(self, ent_id: int):
    return ent_id in self._step_output

class TeamFullyArmed(Task):

  WEAPON_IDS = {
    Action.Melee: {'weapon':5, 'ammo':13}, # Sword, Scrap
    Action.Range: {'weapon':6, 'ammo':14}, # Bow, Shaving
    Action.Mage: {'weapon':7, 'ammo':15} # Wand, Shard
  }

  '''Count the number of fully-equipped agents of a specific skill in the team'''
  def __init__(self, attack_style, min_level: int, num_agent: int):
    assert attack_style in [Action.Melee, Action.Range, Action.Melee], "Wrong style input"
    super().__init__()
    self.attack_style = attack_style
    self.min_level = min_level
    self.num_agent = num_agent

    self.item_ids = { 'hat':2, 'top':3, 'bottom':4 }
    self.item_ids.update(self.WEAPON_IDS[attack_style])
  
  def step(self, gs: GameState):
    super().step(gs)
    data = gs.get_data(ItemState) # 2d numpy data of all the item instances

    flt_idx = (data[:,gs.attr2col(ItemState, 'level')] >= self.min_level) & \
              (data[:,gs.attr2col(ItemState, 'equipped')] > 0)
    
    # should have all hat, top, bottom (general)
    tmp_grpby = {}
    for item, type_id in self.item_ids.items():
      flt_tmp = flt_idx & (data[:,gs.attr2col(ItemState, 'type_id')] == type_id)
      tmp_grpby[item] = \
        self._gs.flt_group_by(data[flt_tmp], gs.attr2col(ItemState, 'owner_id'))

    # get the intersection of all tmp_grpby keys
    equipped_each = [set(equipped.keys()) for equipped in tmp_grpby.values()]
    equipped_all = set.intersection(*equipped_each)

    # aggregate for each team
    for ent_id in equipped_all:
      pop_id = self._gs.ent2pop[ent_id]
      if pop_id in self._step_output:
        self._step_output[pop_id].append(ent_id)
      else:
        self._step_output[pop_id] = [ent_id]

  def evaluate(self, ent_id: int):
    pop_id = self._gs.ent2pop[ent_id]
    if pop_id in self._step_output:
      return self._step_output[pop_id] >= self.num_agent

    return False


class TaskWrapper(nmmo.Env):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # CHECK ME: should every agent have a task assigned?
    # {task: [ent_id, ent_id, ...]}
    self.task_assignment: Dict[Task, List[int]] = {}
    self.ent2task: Dict[int, List[Task]] = {} # reverse map

    # game state generator
    self.gs_gen: GameStateGenerator = None

  def _set_task_assignment(self, task_assignment: Dict[Task, List[int]]):
    self.task_assignment = task_assignment
    self.ent2task = {}
    for task, ent_ids in self.task_assignment.items():
      for ent_id in ent_ids:
        if ent_id in self.ent2task:
          self.ent2task[ent_id].append(task)
        else:
          self.ent2task[ent_id] = [task]

  def reset(self,
            task_assignment: Dict[Task, List[int]],
            map_id=None, seed=None, options=None):
    gym_obs = super().reset(map_id, seed, options)

    self.gs_gen = GameStateGenerator(self.realm, self.config)
    self._set_task_assignment(task_assignment)
    
    return gym_obs

  def _compute_rewards(self, agents: List[AgentID], dones: Dict[AgentID, bool]):
    '''Computes the reward for the specified agent'''
    infos = {}
    rewards = { eid: -1 for eid in dones }

    # CHECK ME: is this a good place to do this?
    gs = self.gs_gen.generate(self.realm, self.obs)
    for task in self.task_assignment:
      task.step(gs)

    for agent_id in agents:
      infos[agent_id] = {}
      agent = self.realm.players.get(agent_id)

      # CHECK ME: can we trust dead agents are not in the agents list?
      if agent is None:
        # assert agent is not None, f'Agent {agent_id} not found'
        rewards[agent_id] = -1
        continue

      # CHECK ME: do we need this?
      infos[agent_id] = {'population': agent.population}

      # CHECK ME: some agents may not have a assinged task. is it ok?
      if agent_id in self.ent2task:
        rewards[agent_id] = sum([task.reward(agent_id) for task in self.ent2task[agent_id]])
        infos[agent_id].update({ str(task): task.evaluate(agent_id)
                                  for task in self.ent2task[agent_id] })
      else:
        # What do we want to do here? Should there be a default task?
        rewards[agent_id] = 0
        infos[agent_id].update({ 'task_assigned': False })

    return rewards, infos


class TestEvalulateTask(unittest.TestCase):

  __test__ = False

  def test_multi_task_eval(self):
    config = ScriptedAgentTestConfig()
    env = TaskWrapper(config)

    # CHECK ME: some agents don't have assigned task. is it ok?
    #   here, agent 1-9 were NOT assigned any tasks
    task_assignment = { LiveLong(): list(range(10, 65)),
                        HoardGold(5): list(range(10, 33)),
                        TeamSizeGE(6): list(range(10, 65)),
                        TeamHoardGold(15): list(range(33, 48)),
                        OwnItem(Item.Ration): list(range(33,65)),
                        EquipItem(Item.Scrap): list(range(10,64,2)),
                        TeamFullyArmed(Action.Melee,1,3): list(range(10,65)) }

    env.reset(task_assignment, seed=RANDOM_SEED)

    for t in range(50):
      obs, rewards, dones, infos = env.step({})

    print(f'{t}: {rewards}')
    print(infos)


if __name__ == '__main__':
  unittest.main()