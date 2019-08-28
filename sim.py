import numpy as np


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

    def __init__(self, initialstate, μstep=np.pi/18, dt=0.01, max_iterations=1000):
        self.allstates = np.empty((max_iterations+1, len(initialstate)))
        self.allstates[0, :] = initialstate
        self.iteration = 0
        self.max_iterations = max_iterations
        self.μstep = μstep
        self.dt = dt

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
        """Do simulation step taking agent's action into account.

        action - 1: decrease bank angle, 2: hold bank angle, 3: increase bank angle
        bank angle is changed by μstep.

        Returns obersvation, reward, done
        """
        self._do_step(action)

        done = self.iteration >= self.max_iterations

        return  self._observe(), self._reward(), done

    def _do_step(self, action:int):
        
        # increment iteration
        self.iteration += 1  

        # unpack last state
        x, y, z, μ, φ = self.laststate

        # calculate thermal velocity
        w = simple_thermal(x, y, z)

        # 
        if action == 0:
            μ_new = μ - self.μstep
        elif action == 1:
            μ_new = μ
        else:
            μ_new = μ + self.μstep

        v = 10
        

        l = v * self.dt

        c_d = 0.03

        z_new = z + w*self.dt - 0.5 * ρ * v**2 * c_d * S * l / (m*g)
        #print(drag(v, μ))

        x_new = x + l * np.cos(μ)
        y_new = y + l * np.sin(μ)

        if μ_new!=0.0:
            r = v**2 / (g*np.tan(μ_new))
            φ_new = φ + l/r
        else:
            φ_new = φ


        self.state = x_new, y_new, z_new, μ_new, φ_new
        

    def _observe(self):
        μ = self.state[-2]
        E_dot = self.state[2] - self.laststate[2] # Hight difference??
        return E_dot, μ

    def _reward(self):
        E = self.state[2] - self.laststate[2] # Hight difference??
        return E
    
    def render(self):
        """Render Simulation state - not implemented yet"""
        pass


def simple_thermal(x, y, z, *args, **kwargs):
    if np.sqrt(x**2 + y**2) < 20:
        return 2.0
    else:
        return 0.0

def thermal(x, y, z, thermal_pos=[0.0, 0.0], z_i=1213, w_star=1.97):
    """Calculate thermal following Allen 2006"""

    from numpy import abs
    if z>0.9*z_i:
        return 0.0
    else:
        r = np.sqrt((x-thermal_pos[0])**2 + (y-thermal_pos[1])**2)
        w_mean = w_star * (z/z_i)**3 * (1 - 1.11 * (z/z_i))
        r2 = np.max(10, 0.102 * (z/z_i)**(1/3) * (1-0.25*(z/z_i)) * z_i)
        r1 = 0.8 if r2<600 else 0.0011*r2 + 0.14
        w_peak = 3 * w_mean * (r2**3 - r2**2*r1) / (r2**3 - r1**3)

        w_D = 0.0 # TODO: calculate

        k1, k2, k3, k4 = _get_ks(r1/r2)
        w = w_peak * ( 1/(1+abs(k1*r/r2+k3)**k2) + k4 * r/r2 + w_D)

        return w


def _get_ks(rr):
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


def drag(v, μ):
    return c0 + c2 * (2*m*g)/(ρ*S*np.cos(μ))


def drag_differential(v, μ):
    return c0 - 4 * c2 * ((2*m*g)/(ρ*S*np.cos(μ)))**2 / v**5
