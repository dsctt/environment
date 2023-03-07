# pylint: disable=all
# This is a prototype. If this direction is correct,
# it will be moved to proper places.

import numpy as np
import numpy_indexed as npi

from testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv

import nmmo
from nmmo.lib.serialized import SerializedState
from nmmo.lib.datastore.numpy_datastore import NumpyDatastore
from nmmo.core.realm import Realm
from nmmo.entity import Entity
from nmmo.systems.item import Item, Hat, Ration
from nmmo.systems import skill as Skill

# pylint: disable=no-member
EventState = SerializedState.subclass("Event", [
  "id",
  "ent_id",
  "population_id",
  "tick",
  
  "event",
  "val_1",
  "val_2",
  "val_3",
])

attr2col = lambda attr: EventState.State.attr_name_to_col[attr]

# EventState.Limits = lambda config: {
#   "id": (0, math.inf),
#   "ent_id": (-math.inf, math.inf),
#   "population_id": (-3, config.PLAYER_POLICIES-1),
#   "tick": (0, math.inf),
#   "event": (0, math.inf),
#   "val_1": (-math.inf, math.inf),
#   "val_2": (-math.inf, math.inf),
#   "val_3": (-math.inf, math.inf),
# }  

class EventCode:
  EAT_FOOD = 1
  DRINK_WATER = 2
  ATTACK = 3
  KILL = 4
  CONSUME = 5
  EQUIP = 6
  PRODUCE = 7
  SELL = 8
  BUY = 9
  GIVE = 10
  EARN_GOLD = 11 # by selling only
  SPEND_GOLD = 12 # by buying only
  GIVE_GOLD = 13

  style2int = { Skill.Melee: 1, Skill.Range:2, Skill.Mage:3 }


class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Event", EventState.State.num_attributes)

    self.event_log = EventLog(self)


# this equals to ItemState.Query
class EventLog(EventCode):
  def __init__(self, realm: Realm):
    self.realm = realm
    self.config = realm.config

    self.datastore = realm.datastore
    self.table = realm.datastore.table('Event')
  
  def reset(self):
    raise NotImplementedError


  # define event logging
  def _create_log(self, entity: Entity, event_code: int):
    log = EventState(self.datastore)
    log.id.update(log.datastore_record.id)    
    log.ent_id.update(entity.ent_id)
    log.population_id.update(entity.population)
    log.tick.update(self.realm.tick)
    log.event.update(event_code)
    
    return log

  def resource(self, entity:Entity, event_code: int):
    assert event_code in [EventCode.EAT_FOOD, EventCode.DRINK_WATER]
    self._create_log(entity, event_code)

  def attack(self, attacker: Entity, style, target: Entity, dmg):
    assert style in self.style2int
    log = self._create_log(attacker, EventCode.ATTACK)
    log.val_1.update(self.style2int[style])
    log.val_2.update(target.ent_id)
    log.val_3.update(dmg)

  def kill(self, attacker: Entity, target: Entity):
    log = self._create_log(attacker, EventCode.KILL)
    log.val_1.update(target.ent_id)
    log.val_2.update(target.population)
    
    # val 3: target level
    # TODO(kywch): attack_level or "general" level?? need to clarify
    log.val_3.update(target.attack_level)

  def item(self, entity: Entity, event_code: int, item: Item):
    assert event_code in [EventCode.CONSUME, EventCode.EQUIP, EventCode.PRODUCE,
                          EventCode.SELL, EventCode.BUY, EventCode.GIVE]
    log = self._create_log(entity, event_code)
    log.val_1.update(item.ITEM_TYPE_ID)
    log.val_2.update(item.level.val)

  def gold(self, entity:Entity, event_code: int, amount: int):
    assert event_code in [EventCode.EARN_GOLD, EventCode.SPEND_GOLD, EventCode.GIVE_GOLD]
    log = self._create_log(entity, event_code)
    log.val_1.update(amount)

  def _get_data(self, event_code):
    data = self.table._data.astype(np.int16)
    flt_idx = (data[:,attr2col('event')] == event_code) \
              & (data[:,0] > 0) # filter out empty records
    return data[flt_idx] # non-empty rows only

  def _flt_group_by(self, flt_data, grpby_col, sum_col=0):
    assert grpby_col in [attr2col(attr) for attr in ['ent_id', 'population_id']], \
          "Invalid group by column"
    assert sum_col in [attr2col(attr) for attr in ['id','val_1','val_2','val_3']], \
          "Invalid sum_col" # cols: id, or val_1-3
    g = npi.group_by(flt_data[:,grpby_col])
    result = {}
    for k, v in zip(*g(flt_data[:,sum_col])):
      if sum_col:
        result[k] = sum(v)
      else:
        result[k] = len(v)
    return result

  def count_group_by(self, event_code, grpby_attr, **kwargs):
    assert grpby_attr in ['ent_id', 'population_id'], "Invalid group by column"
    data = self._get_data(event_code)
    sum_col = attr2col('id') # counting is the default

    if event_code in [EventCode.EAT_FOOD, EventCode.DRINK_WATER]:
      flt_idx = np.ones(data.shape[0], dtype=np.bool_)

    elif event_code == EventCode.ATTACK:
      assert 'style' in kwargs, "style required for attack"
      # log.val_1.update(self.style2int[style])
      flt_idx = self.style2int[kwargs['style']] == data[:,attr2col('val_1')]

    elif event_code == EventCode.KILL:
      assert False, "Define KILL counting spec first"
      # could be npcs only, or with level higher than, or specific foe, etc...

    elif event_code in [EventCode.CONSUME, EventCode.EQUIP, EventCode.SELL,
                        EventCode.PRODUCE, EventCode.BUY, EventCode.GIVE]:
      assert 'item_sig' in kwargs, 'item_sig must be provided'
      assert 2 <= kwargs['item_sig'][0] <= 17, 'Invalid item type'
      assert 0 <= kwargs['item_sig'][1] <= 10, 'Invalid item level'

      # log.val_1.update(item.ITEM_TYPE_ID)
      # log.val_2.update(item.level.val)

      # count the items, the level of which greater than or equal to input level
      flt_idx = (data[:,attr2col('val_1')] == kwargs['item_sig'][0]) \
                & (data[:,attr2col('val_2')] >= kwargs['item_sig'][1])

    elif event_code in [EventCode.EARN_GOLD, EventCode.SPEND_GOLD, EventCode.GIVE_GOLD]:
      flt_idx = np.ones(data.shape[0], dtype=np.bool_)
      # log.val_1.update(amount)
      sum_col = attr2col('val_1') # sum gold amount

    else:
      assert False, "Invalid event code"

    return self._flt_group_by(data[flt_idx], attr2col(grpby_attr), sum_col)


if __name__ == '__main__':
  config =  ScriptedAgentTestConfig()
  env = ScriptedAgentTestEnv(config)
  env.reset()

  env.step({})

  # initialize Event datastore
  env.realm.datastore.register_object_type("Event", EventState.State.num_attributes)

  event_log = EventLog(env.realm)

  # def resource(self, entity:Entity, event_code: int):
  event_log.resource(env.realm.players[1], EventCode.EAT_FOOD)
  event_log.resource(env.realm.players[2], EventCode.DRINK_WATER)

  # def attack(self, attacker: Entity, style, target: Entity, dmg):
  event_log.attack(env.realm.players[2], Skill.Melee, env.realm.players[4], 50)

  # def kill(self, attacker: Entity, target: Entity):
  event_log.kill(env.realm.players[3], env.realm.players[5])

  env.step({})

  # def item(self, entity: Entity, event_code: int, item: Item):
  ration_8 = Ration(env.realm, 8); event_log.item(env.realm.players[4], EventCode.CONSUME, ration_8)
  hat_7 = Hat(env.realm, 7); event_log.item(env.realm.players[5], EventCode.EQUIP, hat_7)
  ration_2 = Ration(env.realm, 2); event_log.item(env.realm.players[6], EventCode.PRODUCE, ration_2)
  ration_3 = Ration(env.realm, 3); event_log.item(env.realm.players[6], EventCode.SELL, ration_3)
  hat_4 = Hat(env.realm, 4); event_log.item(env.realm.players[6], EventCode.BUY, hat_4)
  hat_5 = Hat(env.realm, 5); event_log.item(env.realm.players[7], EventCode.GIVE, hat_5)

  # def gold(self, entity:Entity, event_code: int, amount: int):
  event_log.gold(env.realm.players[8], EventCode.EARN_GOLD, 6)
  event_log.gold(env.realm.players[9], EventCode.SPEND_GOLD, 7)
  event_log.gold(env.realm.players[10], EventCode.GIVE_GOLD, 8)

  print(event_log.count_group_by(EventCode.EAT_FOOD, 'ent_id'))
  print(event_log.count_group_by(EventCode.DRINK_WATER, 'population_id'))
  print(event_log.count_group_by(EventCode.ATTACK, 'ent_id', style=Skill.Melee))

  print(event_log.count_group_by(EventCode.CONSUME, 'ent_id', item_sig=(ration_8.ITEM_TYPE_ID, 5)))
  print(event_log.count_group_by(EventCode.CONSUME, 'ent_id', item_sig=(ration_8.ITEM_TYPE_ID, 9)))

  print(event_log.count_group_by(EventCode.EQUIP, 'ent_id', item_sig=(ration_8.ITEM_TYPE_ID, 9)))
  print(event_log.count_group_by(EventCode.EQUIP, 'ent_id', item_sig=(hat_7.ITEM_TYPE_ID, 5)))

  print(event_log.count_group_by(EventCode.PRODUCE, 'ent_id', item_sig=(ration_2.ITEM_TYPE_ID, 1)))
  print(event_log.count_group_by(EventCode.SELL, 'ent_id', item_sig=(ration_2.ITEM_TYPE_ID, 1)))
  print(event_log.count_group_by(EventCode.BUY, 'ent_id', item_sig=(hat_4.ITEM_TYPE_ID, 1)))
  print(event_log.count_group_by(EventCode.GIVE, 'ent_id', item_sig=(hat_4.ITEM_TYPE_ID, 1)))

  print(event_log.count_group_by(EventCode.EARN_GOLD, 'population_id'))
  print(event_log.count_group_by(EventCode.SPEND_GOLD, 'population_id'))
  print(event_log.count_group_by(EventCode.GIVE_GOLD, 'population_id'))

  print()


