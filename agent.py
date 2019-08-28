import numpy as np
import random

class RandomAgent:
    """A completely random soaring agent.
    This is used to document the api for all agents."""

    def get_action(self, observation):
        """Returns the action the agents intends to do
        at the moment.
        - 0: Decrease bank angle by five degrees.
        - 1: Do nothing.
        - 2: Increase bank angle by five degrees."""
        return random.choice([0,1,2])

    def update(self, observation, action, reward, nextobservation):
        """Does nothing, because this model doesn't learn."""
        pass

class TableAgent:
    """A simple Agent based on Q-Learning.
    Observations are a tuple of:
    1: The derivative of the total energy (kinetic + potential).
    2: The current bank angle.

    1 is also used as the reward.
    When used for indexing the Q-table it is discretized to be either
    rising, neutral or falling."""

    def otos(observation):
        """Convert from observation to indices for our policy."""
        dte, bankangle = observation

        # Discretize dte
        dte = 0 if dte < -self.deadzone else
            2 if dte > self.deadzone else 1

        bankangle = int(bankangle/5 + 9)

    def __init__(self, learning_rate=0.1, discount=0.5,
            randomness=0.3, deadzone=0.1):
        # Learning rate and discount factor are chosen quite randomly
        self.policy = np.zeros((3,9,3))
        self.learning_rate = learning_rate
        self.discount = discount
        self.randomness = randomness
        self.deadzone = deadzone

    def get_action(self, observation):
        if random.random() <= self.randomness:
            return random.choice([0,1,2])

        dte, bankangle = otos(observation)


        # If there is more than one maximum return a random one.
        return random.choice(np.argmax(self.policy[dte, bankangle]))

    def update(self, observation, action, reward, nextobservation):
        dte, bankangle = otos(observation)
        ndte, nbankangle = otos(nextobservation)

        oldq = self.policy[dte, bankangle, action]
        learned = reward + self.discount * (
            np.max(self.policy[ndte, nbankangle]))

        self.policy[dte, bankangle, action] =
            (1-self.learning_rate)*oldq +
            self.learning_rate*learned
