from pdb import set_trace as T
import numpy as np

from tqdm import tqdm
import ray

from neural_mmo.forge.trinity.env import Env
from neural_mmo.forge.trinity.scripted import baselines

from projekt.config import SmallMultimodalSkills, Debug

DEV_AGENTS  = [
      baselines.Hunter, baselines.Fisher, 
      baselines.Prospector, baselines.Carver, baselines.Alchemist,
      baselines.Melee, baselines.Range, baselines.Mage]

config         = SmallMultimodalSkills()
config.AGENTS  = DEV_AGENTS
config.NMOB = 32
config.NENT = 32
config.EVALUTE = True
config.RENDER  = False

@ray.remote
def run_env(worker):
   env = Env(config)
   env.reset()
   for idx  in range(1024):
      if worker == 0 and idx % 10 == 0:
         print(idx)
      env.render()
      env.step({})

   return env.terminal()['Stats']

NUM_CORES = 4
ray.init()
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

   print('{}: {}'.format(key, np.mean(val_ary)))
