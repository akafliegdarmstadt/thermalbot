import sys
from math import sin, cos, tan, pi
import numpy as np

from thermal import thermal


class Normalization:

    def __init__(self, minimal_vector, maximal_vector):
        self.min = minimal_vector
        self.max = maximal_vector

    def __call__(self, value):
        
        if np.any((idx:=value<self.min)):
            self.min[idx] = values[idx]
        elif np.any((idx:=value>self.max)):
            self.max[idx] = values[idx]

        return (value-self.min)/(self.max-self.min)


class Environment:

    observation_length = 5
    action_length = 3
    max_iterations = 10000
    v = 10 # m/s
    c_d = 0.03
    g = 9.81
    ρ = 1.225
    S = 0.1
    m = 0.5

    def __init__(self, initialstate, normalize, dt=0.3, µ_p=np.deg2rad(15)/0.3):
        self.dt = dt
        self.µ_p = µ_p
        self._iteration = 0
        self._statehistory = np.empty((self.max_iterations+1, len(initialstate)))
        self._normalize = normalize

    def do_step(self, action:int):
        """
        Run simulation step taking agent's action into account and return observation

        action - 1: decrease bank angle, 2: hold bank angle, 3: increase bank angle
        bank angle is changed by μstep.

        The observation has the full state first, then it's gradient an then the liftgradient.

        Returns obersvation, reward, done
        """

        self.do_simulation_step(action)

        done = self._iteration >= self.max_iterations

        return self._normalize(self.state), done

    def do_simulation_step(self, action):
        
        self._iteration += 1

        # unpack laststate
        x, y, z, µ, χ = self.laststate

        # calculate thermal velocity
        w = thermal(x, y, z)

        # handle given action
        if action == 0:
            µ -= self.µ_p * self.dt
        elif action == 2:
            µ += self.µ_p * self.dt

        # traveled distance during timestep
        l = self.v * self.dt

        # calculate new airplane position
        x += l * cos(χ)
        y += l * sin(χ)
        z += w * self.dt - 0.5 * self.ρ * self.v**2 * self.c_d * self.S * l / (self.m * self.g)

        # calculate new heading/azim uth angle
        if μ != 0.0:
            r = self.v**2 / (self.g * tan(µ))
            χ = (χ + l/r) % (2 * pi)

        self._statehistory[self._iteration, :] = x, y, z, µ, χ

    @property
    def state(self):
        return self._statehistory[self._iteration, :]

    @property
    def laststate(self):
        return self._statehistory[self._iteration-1, :]


SimpleEnvironment = Environment