from utils import RoomWorldMDP
import numpy as np

mdp = RoomWorldMDP()

# Because the reward is so small, to prevent miniscule numbers, I add a constant growth factor
r_growth = 1000

# initialize value map
v_map = np.full(mdp.states.shape, 0)
# Assure that V(terminal) = 0
for s in mdp.terminal:
    v_map[s] = 0

theta = 1e-8
delta = np.inf
###
# Policy Evauluation
###
while delta >= theta:
    delta = 0
    for s in mdp.states:
        old_v = v_map[s]
        act_vals = [(mdp.transition_dynamics[s,a,:] * (mdp.rewards[s,a]*r_growth + mdp.gamma * v_map)).sum() for a in mdp.actions]
        v_map[s] = np.array(act_vals).max()
        delta = max(delta, abs(old_v - v_map[s]))

###
# Policy Improvement
###
policy = np.full(v_map.shape, -1)

for s in mdp.states:

    # pick best action
    act_vals = [(mdp.transition_dynamics[s,a,:] * (mdp.rewards[s,a]*r_growth + mdp.gamma * v_map)).sum() for a in mdp.actions]
    best_action = np.argmax(np.array(act_vals))
    # update policy
    policy[s] = best_action

mdp.pprint_policy(policy)