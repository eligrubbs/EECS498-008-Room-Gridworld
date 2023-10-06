from utils import RoomWorldMDP, get_random_policy
import numpy as np


# Because the reward is so small, to prevent miniscule numbers, I add a constant growth factor
mdp = RoomWorldMDP(reward_scaler=1000)

# initialize
v_map = np.full(mdp.states.shape, 0)
pi = get_random_policy(stochastic=False)

# params
theta = 1e-8
delta = np.inf
policy_stable = False

iters = 0

while not policy_stable:
    iters += 1
# Poliy Evaluation
    delta = 0
    for s in mdp.states:
        old_v = v_map[s]

        act_val = (mdp.transition_dynamics[s,pi[s],:] * (mdp.rewards[s,pi[s]] + mdp.gamma * v_map)).sum()
        v_map[s] = act_val

        delta = max(delta, abs(old_v - v_map[s]))
    
# Policy Improvement
    policy_stable = True
    for s in mdp.states:
        old_a = pi[s]

        act_vals = [(mdp.transition_dynamics[s,a,:] * (mdp.rewards[s,a] + mdp.gamma * v_map)).sum() for a in mdp.actions]
        pi[s] = np.argmax(np.array(act_vals))

        if old_a != pi[s]:
            policy_stable = False


print(f"Iterations until convergence: {iters}")
mdp.pprint_policy(pi)
