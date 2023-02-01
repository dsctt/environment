import nmmo


def test_rollout():
   config = nmmo.config.Default()  
   config.PLAYERS = [nmmo.core.agent.Random]

   env = nmmo.Env(config)
   env.reset()
   for i in range(128):
       env.step({})

if __name__ == '__main__':
   test_rollout()
