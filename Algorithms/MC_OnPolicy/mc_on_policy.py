from utils import *
import numpy as np
from tqdm import tqdm

env = RoomWorldEnv(reward_scaler=1000)

# parameters
epsilon = 0.3
iterations = 5000

# initialization
pi = get_random_policy()
pi = get_ep_soft_policy(pi, epsilon=epsilon)

q_map = np.full((121,4), 0)

returns = {(s,a): [] for s in env.observation_space for a in env.action_space}

for itr in tqdm(range(iterations)):
    # Generate episode
    episode = generate_episode(env, pi)
    ep_len = len(episode)
    g = 0

    # backwards through episode
    for i, (s, a, r) in enumerate(reversed(episode)):
        # Policy Evaluation
        g = (env.gamma * g) + r

        if (s,a) not in [(s,a) for s,a,_ in episode[:(ep_len-i-1)]]:
            returns[(s,a)].append(g)
            q_map[s,a] = np.mean(returns[(s,a)])
        # Policy Iteration
            a_star = q_map[s,:].argmax()
            probs = [epsilon/4]*4
            probs[a_star] += 1-epsilon
            pi[s,:] = probs

env.pprint_policy(pi)
