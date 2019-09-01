from math import sin, cos, tan,  pi
import numpy as np
import numba

from .thermal import thermal

# 
g = 9.81
# air density
ρ = 1.225
# airplane mass
m = 0.5
# wing area
S = 0.1

# lilienthal polar constants c0+c_l*c2
c0 = 0.06
c2 = 0.04


class SimpleFDM:

    c_d = 0.03

    def __init__(self, pos=[.0, .0, .0], v=10.0, dt=0.01, thermal=thermal):
        self.pos = pos
        self.v = v
        self.μ = 0.0
        self.μstep = 5.0
        self.φ = 0.0
        self.time = 0.0
        self.dt = dt
        self.thermal = thermal

    def run(self, action:int):
        if action == 0:
            self.μ = self.μ - self.μstep
        elif action == 2:
            self.μ = self.μ + self.μstep
        
        l = self.v * self.dt

        w = self.thermal(*self.pos)

        self.pos[0] += l * cos(self.φ)
        self.pos[1] += l * sin(self.φ)
        self.pos[2] += w*self.dt - 0.5 * ρ * self.v**2 * self.c_d * S * l / (m*g)

        if self.μ != 0.0:
            r = self.v**2 / (g*np.tan(self.μ))
            self.φ = (self.φ + l/r)%(2*pi)

        self.time += self.dt

        return (*self.pos, self.v, self.μ, self.φ, self.time)


class Environment:
    """A thermal gliding environment.
    """

    @classmethod
    def generate_state(x, y, z, v, μ, φ, t):
        """Pack state variables into state vector
        """
        return np.array([x, y, z, v, μ, φ, t])

    def __init__(self, initialstate, fdm=SimpleFDM(), max_iterations=1000):
        self.allstates = np.empty((max_iterations+1, len(initialstate)))
        self.allstates[0, :] = initialstate
        self.iteration = 0
        self.max_iterations = max_iterations

        self.fdm = fdm
        # initiate
        self.fdm.pos = initialstate[:3]

        # TODO: remove random choosen diff bounds
        self.upper_bounds = [5000, 5000, 2000.0, 100.0, pi/2, pi, 100]
        self.lower_bounds = [-5000, -5000, 0.0, 0.0, -pi/2, -pi, 0]
        self.diff_upper_bounds = [100, 100, 100, 100, 10, g*tan(10)*0.01/100**2, 100]
        self.diff_lower_bounds = -np.array(self.diff_upper_bounds)


    @property
    def laststate(self):
        if self.iteration>0:
            return self.allstates[self.iteration-1]
        else:
            return None

    @property
    def state(self):
        return self.allstates[self.iteration]
    
    @state.setter
    def state(self, newstate):
        self.allstates[self.iteration] = newstate

    @property
    def dstate(self):
        return (self.state-self.laststate)/self.fdm.dt

    @property
    def full_state(self):
        norm_state = [_interp(statevar, self.lower_bounds[i], self.upper_bounds[i]) for i, statevar in enumerate(self.state)]

        norm_dstate = [_interp(statevar, self.diff_lower_bounds[i], self.diff_upper_bounds[i]) for i, statevar in enumerate(self.dstate)]

        return norm_state + norm_dstate

    def step(self, action:int):
        """Do simulation step taking agent's action into account.

        action - 1: decrease bank angle, 2: hold bank angle, 3: increase bank angle
        bank angle is changed by μstep.

        Returns obersvation, reward, done
        """

        # increment iteration
        self.iteration += 1  

        self.state = self.fdm.run(action)

        done = self.iteration >= self.max_iterations

        liftgradient = self.get_liftgradient(self.state)
        observation = self.full_state + [liftgradient]

        return observation, done

    def get_liftgradient(self, state):
        x0, y0, z = state[:3]
        phi = state[4]

        x1, y1 = x0 + sin(phi), y0 + cos(phi)

        l0 = thermal(x0, y0, z)
        l1 = thermal(x1, y1, z)

        return l1-l0    


def _interp(x, xp1, xp2, yp1=0, yp2=1):
    if x < xp1:
        return yp1
    elif x > xp2:
        return yp2

    return (x-xp1)/(xp2-xp1) * (yp2-yp1)



