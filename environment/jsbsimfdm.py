import os
from .thermal import thermal
import jsbsim


class JSBsimFDM:
    def __init__(self, initialstate):
        # create jsbsim instance
        self.fdm = jsbsim.FGFDMExec(os.environ['JSBsim_PATH'], None)

        # load random script
        self.fdm.load_script('scripts/c3105.xml')

        # do initial step
        self.fdm.run_ic()

    def set_initial(self, pos, v):
        pass

    def run(self, action:int):
        # get position and time
        x, y, z = 0, 0, 0

        # update wind
        w = thermal(x, y, z)

        self.fdm['atmosphere/wind-down-fps'] = w # TODO: conversation to feet per second needed

        # do simulation step
        self.fdm.run()        

        # return current state
        return None