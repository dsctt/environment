from pdb import set_trace as T
import numpy as np

from tqdm import tqdm
import ray

from neural_mmo.forge.trinity.env import Env
from neural_mmo.forge.trinity.scripted import baselines

from projekt.config import SmallMultimodalSkills, Debug

DEV_AGENTS  = [
      baselines.Fisher, baselines.Herbalist, 
      baselines.Prospector, baselines.Carver, baselines.Alchemist,
      baselines.Melee, baselines.Range, baselines.Mage]

config         = SmallMultimodalSkills()
config.AGENTS  = DEV_AGENTS
config.NMOB    = 64
config.NENT    = 32
config.EVALUTE = True
config.RENDER  = True

HORIZON = 1024
if config.RENDER:
   HORIZON = 999999999

@ray.remote
def run_env(worker):
   env = Env(config)
   env.reset()
   for idx  in range(HORIZON):
      if worker == 0 and idx % 10 == 0:
         print(idx)
      env.render()
      env.step({})

   return env.terminal()['Stats']

NUM_CORES = 1
ray.init(local_mode=True)
results = []
for worker in range(NUM_CORES):
   result = run_env.remote(worker)
   results.append(result)
results = ray.get(results)

key_packet = results[0]
for key in key_packet:
   val_ary = []
   for i in range(NUM_CORES):
      vals = results[i][key]
      val_ary.append(np.mean(vals))

   print('{0:>40}'.format(key), ':', np.mean(val_ary))


