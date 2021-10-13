from pdb import set_trace as T

from neural_mmo.forge.trinity.env import Env

from projekt.config import SmallMultimodalSkills

config         = SmallMultimodalSkills()
config.AGENTS  = config.DEV_AGENTS
config.EVALUTE = True
config.RENDER  = True

env = Env(config)

env.reset()
while True:
   env.render()
   env.step({})

