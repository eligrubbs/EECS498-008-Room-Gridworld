from .mdp import RoomWorldMDP
from .simulator import RoomWorldEnv
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
        if policy.shape == (121,):
            probs[policy[s]] += 1-epsilon
        else:
            probs[policy[s].argmax()] += 1-epsilon
        return probs
    new_policy = np.array([ep_function(s) for s in range(121)])

    return new_policy

def ep_soft_from_q_map(q_map: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """
    Create and return an epsilon soft policy for the RoomWorldMDP/Env from a state-action value map.

    Args:
        q_map: numpy array with shape (121,4) which holds values for each state, action pair
        epislon: exploratory probability.

    Returns:
        new_policy: epsilon-greedy policy w.r.t original q_map and epsilon
                    np array of size (121,4)
    """
    greedy_policy = q_map.argmax(axis=1)
    new_policy = get_ep_soft_policy(greedy_policy, epsilon=epsilon)
    return new_policy

def generate_episode(env: RoomWorldEnv, policy) -> list[tuple[int, int, float]]:
    """
    Generate an episode in the RoomWorldEnv by executing the passed in policy.

    Args:
        env: RoomWorldEnv object
        policy: A valid policy for the RoomWorldMDP/Env
                np array of shape (121,4) or (121,) if deterministic

    Returns:
        episode: list of (s_t, a_t, r_t+1) tuples where
                 s_t: state of the world at time t
                 a_t: action taken at time t
                 r_t+1: reward earned at time t+1
    """
    episode = []

    s, _ = env.reset()

    while True:
        a = sample_policy(policy, s)
        s_p, r, terminated, truncated, _ = env.step(a)
        
        episode.append((s, a, r))
        s = s_p

        if terminated or truncated:
            break

    return episode

def sample_policy(policy: np.ndarray, state: int) -> int:
    """
    Sample the passed in policy.

    Args:
        policy: policy to sample from
                np array of shape (121,4) or (121,) if deterministic
        state: integer representing current state an action must be taken in
    
    Returns:
        action: integer representing which action to take
    """
    if policy.shape == (121,4):
        return int(np.random.choice(4,1,p=policy[state]))
    return int(policy[state])
