import numpy as np
import random

class RandomAgent:
    """A completely random soaring agent.
    This is used to document the api for all agents."""
    current_action = 1

    def get_action():
        """Returns the action the agents intends to do
        at the moment."""
        return self.current_action

    def update(state, action, reward):
        """Chooses a new action to do from [0,1,2].
        - 0: Decreases roll angle.
        - 1: Does nothing.
        - 2: Increases roll angle."""
        current_action = random.choice([0,1,2])
