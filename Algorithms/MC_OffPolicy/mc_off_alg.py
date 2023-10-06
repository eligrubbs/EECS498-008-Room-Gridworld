from utils import *
import numpy as np
from tqdm import tqdm


env = RoomWorldEnv(reward_scaler=1000)

# parameters
iterations = 100000

# initialize
q_map = np.full((121,4), 0)
c_map = np.full((121,4), 0)

pi = np.array([q_map[s,:].argmax() for s in env.observation_space])

for itr in tqdm(range(iterations)):
    b = get_ep_soft_policy(get_random_policy(stochastic=False))
    # Generate episode
    episode = generate_episode(env, b)
    ep_len = len(episode)
    g = 0
    w = 1

    # backwards through episode
    for s, a, r in reversed(episode):
        # Eval
        g = (env.gamma * g) + r
        c_map[s,a] = c_map[s,a] + w
        q_map[s,a] = q_map[s,a] + ( (w/c_map[s,a]) * (g - q_map[s,a]) )
        # Control
        pi[s] = q_map[s,:].argmax()
        if a != pi[s]:
            break
        w  = w * 1/(b[s,a])

env.pprint_policy(pi)
