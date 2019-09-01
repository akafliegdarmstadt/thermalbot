import jsbsim


class JSBsimFDM:
    def __init__(self, initialstate):
        jsbsim.FGFDMExec(jsbsimpath, None)

        jsbsim.load_script(scriptpath)

        jsbsim.run_ic()

    def set_initial(self, pos, v):
        pass

    def run(self, action:int):
        jsbsim.run()