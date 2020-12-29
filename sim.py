from math import sin, cos, tan, pi
import numpy as np

from thermal import thermal

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


class Simulation:
    """A thermal gliding simulation.
    """

    @classmethod
    def generate_state(x, y, z, μ, φ):
        """Pack state variables into state vector
        """
        return np.array([x, y, z, μ, φ])

    def __init__(
        self, initialstate, μstep=(15 * np.pi / 360), dt=0.01, max_iterations=1000
    ):
        self.allstates = np.empty((max_iterations + 1, len(initialstate)))
        self.allstates[0, :] = initialstate
        self.iteration = 0
        self.max_iterations = max_iterations
        self.μstep = μstep
        self.dt = dt

        self.v = 10
        self.upper_bounds = [5000, 5000, 2000, pi / 2, pi]
        self.lower_bounds = [-5000, -5000, 0, -pi / 2, -pi]
        self.diff_upper_bounds = [
            self.v,
            self.v,
            self.v,
            μstep / dt,
            g * tan(μstep) * dt / self.v ** 2,
        ]
        self.diff_lower_bounds = -np.array(self.diff_upper_bounds)

    @property
    def laststate(self):
        if self.iteration > 0:
            return self.allstates[self.iteration - 1]
        else:
            return None

    @property
    def state(self):
        return self.allstates[self.iteration]

    @state.setter
    def state(self, newstate):
        """ Returns the current state. """
        self.allstates[self.iteration] = newstate

    @property
    def dstate(self):
        """ Returns the gradient of the current state """
        return (self.state - self.laststate) / self.dt

    @property
    def full_state(self):
        """ Returns the normalised state and the normalised dstate. """
        norm_state = [
            _interp(statevar, self.lower_bounds[i], self.upper_bounds[i])
            for i, statevar in enumerate(self.state)
        ]

        norm_dstate = [
            _interp(statevar, self.diff_lower_bounds[i], self.diff_upper_bounds[i])
            for i, statevar in enumerate(self.dstate)
        ]

        return norm_state + norm_dstate

    def step(self, action: int):
        """Do simulation step taking agent's action into account.

        action - 1: decrease bank angle, 2: hold bank angle, 3: increase bank angle
        bank angle is changed by μstep.

        The observation has the full state first, then it's gradient an then the liftgradient.

        Returns obersvation, reward, done
        """
        self._do_step(action)

        done = self.iteration >= self.max_iterations

        b = 1.0
        liftgradient = self.get_liftgradient(self.full_state)

        observation = np.concatenate([self.state, self.dstate, [liftgradient]])

        return observation, done

    def _do_step(self, action: int):

        # increment iteration
        self.iteration += 1

        # unpack last state
        x, y, z, μ, φ = self.laststate

        # calculate thermal velocity

        w = thermal(x, y, z) * 10

        #
        if action == 0:
            μ_new = μ - self.μstep
        elif action == 1:
            μ_new = μ
        else:
            μ_new = μ + self.μstep

        μ_new = np.clip(μ_new, np.deg2rad(-45), np.deg2rad(45))
        v = self.v

        l = v * self.dt

        c_d = 0.03

        z_new = z + w * self.dt - 0.5 * ρ * v ** 2 * c_d * S * l / (m * g)

        x_new = x + l * cos(φ)
        y_new = y + l * sin(φ)

        if μ_new != 0.0:
            r = v ** 2 / (g * np.tan(μ_new))
            φ_new = (φ + l / r) % (2 * pi)
        else:
            φ_new = φ

        self.state = x_new, y_new, z_new, μ_new, φ_new

    def get_liftgradient(self, state):
        x0, y0, z = state[:3]
        phi = state[4]

        x1, y1 = x0 + sin(phi), y0 + cos(phi)

        l0 = thermal(x0, y0, z)
        l1 = thermal(x1, y1, z)

        return l1 - l0


def _interp(x, xp1, xp2, yp1=0, yp2=1):
    if x < xp1:
        return yp1
    elif x > xp2:
        return yp2

    return (x - xp1) / (xp2 - xp1) * (yp2 - yp1)


def simple_thermal(x, y, z, *args, **kwargs):
    if np.sqrt(x ** 2 + y ** 2) < 20:
        return 2.0
    else:
        return 0.0


def drag(v, μ):
    return c0 + c2 * (2 * m * g) / (ρ * S * np.cos(μ))


def drag_differential(v, μ):
    return c0 - 4 * c2 * ((2 * m * g) / (ρ * S * np.cos(μ))) ** 2 / v ** 5
