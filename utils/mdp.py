
import numpy as np
import numpy.testing as npt

class RoomWorldMDP:
    """
    MDP implementaion of the Gridworld.
    Since the definitions deal only with sets, some liberty was taken when
    labeling each state so that semantically it lines up better with the picture.

    State arrangement:  
    The outer shaded squares are not considered. The inner box of 121 spaces  
    are labeled in standard reading order, with shaded squares ALSO being labeled.  
    
    This is so the end of every row ends in a multiple of 11 minus 1 (zero indexed), making our lives much easier. 
    Some shaded states fall into the 121 that we consider. They will be inaccessible states.
    """

    def __init__(self, gamma:float = 0.9):
        """
        Initialize the Room Gridworld.

        Args:
            gamma: discount rate for the world.
                   default: 0.9
        """
        self.hallways = [27,56,74,104]
        self.terminal = [96,104]
        self.inaccessible = [5,16,38,49,60,71,82,93,115,
                                      55,57,58,59,
                                      72,73,75,76]

        self.gamma = gamma

        self.states = np.arange(121)

        # Semantically
        #           0: up
        #     3: left   1: right
        #          2: down
        # Will not matter that the terminal/inaccessible states have actions
        # Because the transition dynamics will take them to themselves always
        self.actions = np.arange(4)

        self.rewards = np.full((121,4), 0)
        # (s,a) that lead to 96
        self.rewards[107,0] = 1
        self.rewards[95,1] = 1
        self.rewards[85,2] = 1
        self.rewards[97,3] = 1
        # (s,a) that lead to 104
        self.rewards[103,1]= 1
        self.rewards[105,3]= 1

        self.transition_dynamics = np.full((121,4,121), 0, dtype=np.float32)
        self.change = {0:-11, 1:1, 2: 11, 3:-1}
        for s in range(121):
            # If terminal or inaccessible
            if s in self.terminal or s in self.inaccessible:
                self.transition_dynamics[s,:,s] = 1 # it broadcasts 1 to all actions

            # Prevent going out of bounds
            if s % 11 == 0: # can't go left
                self.transition_dynamics[s,3,s] = 1
            if s < 11: # can't go up
                self.transition_dynamics[s,0,s] = 1
            if s > 109: # can't go down
                self.transition_dynamics[s,2,s] = 1
            if s % 10 == 0 and s != 0: # can't go right
                self.transition_dynamics[s,1,s] = 1

            for a in range(4):
                if self.transition_dynamics[s,a,s] == 1:
                    continue
                # Every action fails 33% of the time
                self.transition_dynamics[s,a,s] = 1/3

                s_p = self.change[a] + s
                # Do not allow navigation to an inaccessible state
                if s_p in self.inaccessible:
                    self.transition_dynamics[s,a,s] = 1
                else:
                    # Else you can navigate their when taking this action 66% of time
                    self.transition_dynamics[s,a,s_p] = 2/3


    def get_MDP(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Return the MDP as a 5-tuple

        Returns:
            states: array states of the environment
                    np.ndarray of shape (121,)
            actions: array of possible actions
                    np.ndarray of shape (4,)
            transition_dynamics: array of probability of going to s' given s,a
                    np.ndarray of shape (121,4,121)
            rewards: array of reward obtained when executing an (s,a) pair
                    np.ndarray of shape (121,4)
        """

        return (self.states, self.actions, self.transition_dynamics, self.rewards, self.gamma)


    def is_inaccessible(self, state: int) -> bool:
        """
        Return the truth value as to whether this state is an inaccessible one.
        """
        return state in self.inaccessible


    def is_terminal(self, state: int) -> bool:
        """
        Determine whether passed in state is a terminal state.
        """
        return state in self.terminal


    def is_hallway(self, state: int) -> bool:
        """
        Determine whether passed in state is a hallway.
        """
        return state in self.hallways

if __name__=="__main__":
    # Test the creation of the MDP

    mdp = RoomWorldMDP()
    
    # test the dynamics and that they are valid
    p = mdp.transition_dynamics
    npt.assert_equal(p.sum(axis=2), np.full((121,4), 1))

    print(mdp.actions.shape)
