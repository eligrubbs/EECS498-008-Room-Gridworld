from utils import *
import numpy as np
from tqdm import tqdm

env = RoomWorldEnv(reward_scaler=1000)

# parameters
epsilon = 0.3
alpha = 0.2
episodes = 1000

q_map = np.full((121,4), 0)

for ep in range(episodes):
    s, _ = env.reset()
    a = np.random.choice(4, p=ep_soft_from_q_map(q_map)[s])
    while True:
        s_p, reward, terminated, truncated, _ = env.step(a)
        a_p = np.random.choice(4, p=ep_soft_from_q_map(q_map)[s_p])

        q_map[s,a] = q_map[s,a] + alpha * (reward + env.gamma*q_map[s_p,a_p] - q_map[s,a])
        if terminated or truncated:
            break
        s = s_p
        a = a_p

env.pprint_policy(ep_soft_from_q_map(q_map))