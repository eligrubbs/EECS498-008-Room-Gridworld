from .mdp import RoomWorldMDP
import numpy as np

def get_random_policy(stochastic: bool = True) -> np.ndarray:
    """
    Create and return a random policy for the RoomWorldMDP/Env.

    Args:
        stochastic: bool indicating whether the policy should be stochatis or deterministic.

    Returns:
        policy: mapping of states to actions
                np array of size (121,4) if stochastic
                np array of size (121,) if deterministic
    """
    def policy_function():
        logits = np.random.random(4)
        if stochastic:
            policy = logits / logits.sum()
        else:
            policy = logits.argmax()
        return policy
    policy = np.array([policy_function() for i in range(121)])

    return policy

def get_ep_soft_policy(policy: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Create and return an epsilon soft policy for the RoomWorldMDP/Env.

    Args:
        policy: policy to turn into epsilon soft policy.
                np array of size (121,4) or (121,) if deterministic
        epislon: exploratory probability.

    Returns:
        new_policy: epsilon-greedy policy w.r.t original policy and epsilon
                    np array of size (121,4)
    """
    def ep_function(s):
        probs = np.array([epsilon/4]*4)
        probs[policy[s].argmax()] += 1-epsilon
        return probs
    new_policy = np.array([ep_function(s) for s in range(121)])

    return new_policy
