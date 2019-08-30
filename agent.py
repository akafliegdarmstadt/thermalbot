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

    def otos(self, observation):
        """Convert from observation to indices for our policy."""
        x, y, z, bankangle, phi, dx, dy, dz, dbankangle, dphi = observation
        
        dy = int(np.round(dy*2))
        bankangle = int(np.round(bankangle*18))

        return dy, bankangle

    def __init__(self, learning_rate=0.1, discount=0.9,
            randomness=0.3, deadzone=0.1):
        # Learning rate and discount factor are chosen quite randomly
        self.policy = np.zeros((3,19,3))
        self.learning_rate = learning_rate
        self.discount = discount
        self.randomness = randomness
        self.randomnessdecay = 1
        self.deadzone = deadzone

    def get_action(self, observation):
        if random.random() <= self.randomness:
            return random.choice([0,1,2])

        dy, bankangle = self.otos(observation)

        # If there is more than one maximum return a random one.
        return np.argmax(self.policy[dy, bankangle])

    def update(self, observation, action, reward, nextobservation):
        dy, bankangle = self.otos(observation)
        ndy, nbankangle = self.otos(nextobservation)

        oldq = self.policy[dy, bankangle, action]
        learned = reward + self.discount * (
            np.max(self.policy[ndy, nbankangle]))

        self.policy[dy, bankangle, action] = \
            (1-self.learning_rate)*oldq + \
            self.learning_rate*learned

        self.randomness *= self.randomnessdecay


class SARSAAgent:
    def otos(self, observation):
        """Convert from observation to indices for our policy."""
        x, y, z, bankangle, phi, dx, dy, dz, dbankangle, dphi, liftgradient = observation
        
        dy = int(np.round(dy*2))
        bankangle = int(np.round(bankangle*18))

        lg = 1
        if liftgradient < -self.deadzone:
            lg = 0
        elif liftgradient > self.deadzone:
            lg = 2

        return dy, bankangle, lg

    def __init__(self, learning_rate=0.9, discount=0.0, randomness=0.3
            deadzone=0.1):
        # Learning rate and discount factor are chosen quite randomly
        self.policy = np.zeros((3,19,3))
        self.learning_rate = learning_rate
        self.discount = discount
        self.deadzone = deadzone
        self.randomness = randomness

    def get_action(self, observation):
        if random.random() <= self.randomness:
            return random.choice([0,1,2])
        
        dte, bankangle, lg = self.otos(observation)

        # If there is more than one maximum return a random one.
        return np.argmax(self.policy[dte, bankangle, lg])

    def update(self, observation, action, reward, nextobservation):
        dy, bankangle, lg = self.otos(observation)
        ndy, nbankangle, nlg = self.otos(nextobservation)
        
        nextaction = self.get_action(nextobservation)

        thisq = self.policy[dy, bankangle, lg]
        nextq = self.policy[ndy, nbankangle, nlg]

        self.policy[dy, bankangle, lg] = \
            thisq + self.learning_rate*(reward + self.discount*nextq - thisq)
