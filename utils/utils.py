from .mdp import RoomWorldMDP
import numpy as np

def get_random_policy(stochastic: bool = True) -> np.ndarray:
    """
    Create and return a random policy for the RoomWorldMDP.

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
