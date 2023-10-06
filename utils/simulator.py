from .mdp import RoomWorldMDP
import numpy as np
# import pygame


class RoomWorldEnv(RoomWorldMDP):
    metadata = {"render_modes": ["human", "array"], "render_fps": 4}

    def __init__(self, max_steps:int = 50, render_mode:str = None, **kwargs):
        """
        Simulation Environment for Room World Environment.

        """
        super().__init__(**kwargs)

        self.max_steps = max_steps
        
        self.observation_space = self.states

        self.action_space = self.actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None) -> tuple[int, dict]:
        """
        Reset the environment.
        """
        np.random.seed(seed=seed)

        self.steps = 0

        starting_spaces = self.accessible
        self.position = int(np.random.choice(starting_spaces, size=1)[0])


        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """
        Take a step in the environemnt by executing action at the current state.
        """

        probs = self.transition_dynamics[self.position, action, :]
        new_pos = int(np.random.choice(a=121,size=1,p=probs))

        reward = self.rewards[self.position, action, new_pos]

        self.steps += 1
        
        truncated = True if self.steps >= self.max_steps else False
        terminated = True if new_pos in self.terminal else False

        self.position = new_pos

        info = self._get_info()
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> int:
        """
        Helper function for returning the observation at each step.
        """
        return self.position

    def _get_info(self) -> dict:
        """
        Return custom information about the environment at each step.
        """
        return {}

    def close(self):
        """
        Stub for when a visual component is added to the environment.
        """
        return


if __name__=="__main__":
    # Simple testing

    env = RoomWorldEnv(max_steps=10)

    obs, _ = env.reset()
    print(obs)
    while True:
        action = np.random.choice(env.actions)
        obs, reward, term, trunc, info = env.step(action)
        print(obs)
        if term or trunc:
            break
