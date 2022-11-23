import unittest
import nmmo.lib.task as task
from nmmo.core.realm import Realm

class Success(task.Task):
  def completed(self, realm: Realm) -> bool:
    return True
  
class Failure(task.Task):
  def completed(self, realm: Realm) -> bool:
    return False
  
class TestTask(task.TargetTask):
  def __init__(self, target: task.TaskTarget, param1: int, param2: float) -> None:
    super().__init__(target)
    self._param1 = param1
    self._param2 = param2

  def completed(self, realm: Realm) -> bool:
    return False
  
  def to_string(self) -> str:
    return f"(TestTask {self._target.to_string()} {self._param1} {self._param2})"

class MockRealm(Realm):
  def __init__(self):
    pass

realm = MockRealm()

class TestTasks(unittest.TestCase):

    def test_operators(self):
      self.assertFalse(task.AND(Success(), Failure(), Success()).completed(realm))
      self.assertTrue(task.OR(Success(), Failure(), Success()).completed(realm))
      self.assertTrue(task.AND(Success(), task.NOT(Failure()), Success()).completed(realm))

    def test_strings(self):
      self.assertEqual(task.AND(Success(), task.NOT(task.OR(Success(), Failure()))).to_string(),
      "(AND Success (NOT (OR Success Failure)))"
      )

    def test_team_helper(self):
      team_helper = task.TeamHelper(range(1, 101), 5)
 
      self.assertSequenceEqual(team_helper.own_team(17).agents(), range(1, 21))
      self.assertSequenceEqual(team_helper.own_team(84).agents(), range(81, 101))

      self.assertSequenceEqual(team_helper.left_team(84).agents(), range(61, 81))
      self.assertSequenceEqual(team_helper.right_team(84).agents(), range(1, 21))

      self.assertSequenceEqual(team_helper.left_team(17).agents(), range(81, 101))
      self.assertSequenceEqual(team_helper.right_team(17).agents(), range(21, 41))

      self.assertSequenceEqual(team_helper.all().agents(), range(1, 101))

    def test_task_target(self):
      tt = task.TaskTarget("Foo", [1, 2, 8, 9])

      self.assertEqual(tt.member(2).to_string(), "Foo.2")
      self.assertEqual(tt.member(2).agents(), [8])

    def test_sample(self):
      sampler = task.TaskSampler()

      sampler.add_task_spec(Success)
      sampler.add_task_spec(Failure)
      sampler.add_task_spec(TestTask, [
        [task.TaskTarget("t1", []), task.TaskTarget("t2", [])],
        [1, 5, 10],
        [0.1, 0.2, 0.3, 0.4]
      ])

      sampler.sample(max_clauses=5, max_clause_size=5, not_p=0.5)

    def test_default_sampler(self):
      team_helper = task.TeamHelper(range(1, 101), 5)
      sampler = task.TaskSampler.create_default_task_sampler(team_helper, 10)

      sampler.sample(max_clauses=5, max_clause_size=5, not_p=0.5)

if __name__ == '__main__':
    unittest.main()