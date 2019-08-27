import numpy as np


class Simulation:

    @classmethod
    def generate_state(x, y, z, μ):
        return np.array([x, y, z, μ])

    def __init__(self, initialstate, μstep=10, max_iterations=1000):
        self.allstates = np.empty((len(initialstate, max_iterations+1)))
        self.allstates[0, :] = initialstate
        self.iteration = 0
        self.max_iterations = max_iterations
        self.μstep = 10

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

    def step(self, action:int):
        self._do_step(action)

        done = self.iteration >= self.max_iterations

        return  self._observe(), self._reward(), done

    def _do_step(self, action:int):
        
        self.iteration += 1  

        x, y, z, μ, φ = self.laststate

        w = thermal(x, y, z)

        if action == 0:
            μ_new = μ - self.μstep
        elif action == 1:
            μ_new = μ
        else:
            μ_new = μ + self.μstep

        # TODO: calculate new position and new heading
        # self.state = ...

    def _observe(self):
        μ = self.state[-2]
        E = self.state[2] - self.laststate[2] # Hight difference??
        return 

    def _reward(self):
        E = self.state[2] - self.laststate[2] # Hight difference??
        return E
    
    def render(self):
        pass


def thermal(x, y, z, thermal_pos=[0.0, 0.0], z_i=1213, w_star=1.97):
    from numpy import abs
    if z>0.9*z_i:
        return 0.0
    else:
        w_D = 0.0 # TODO: calculate
        r = np.sqrt((x-thermal_pos[0])**2 + (y-thermal_pos[1])**2)
        w_mean = w_star * (z/z_i)**3 * (1 - 1.11 * (z/z_i))
        r2 = np.max(10, 0.102 * (z/z_i)**(1/3) * (1-0.25*(z/z_i)) * z_i)
        r1 = 0.8 if r2<600 else 0.0011*r2 + 0.14
        w_peak = 3 * w_mean * (r2**3 - r2**2*r1) / (r2**3 - r1**3)

        k1, k2, k3, k4 = get_ks(r1/r2)
        w = w_peak * ( 1/(1+abs(k1*r/r2+k3)**k2) + k4 * r/r2 + w_D)

        return w

def get_ks(rr):
    from scipy.interpolate import interp1d

    rrs = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
    ks = np.array([[1.5352, 2.5826, -0.0113, 0.0008],
                   [1.5265, 3.6054, -0.0176, 0.0005],
                   [1.4866, 4.8354, -0.0320, 0.0001],
                   [1.2042, 7.7904, 0.0848, 0.0001],
                   [0.8816, 13.972, 0.3404, 0.0001],
                   [0.07067, 23.994, 0.5689, 0.0002],
                   [0.6189, 42.797, 0.7157, 0.0001]])

    return interp1d(rrs, ks, kind='nearest', axis=0)(rr)
