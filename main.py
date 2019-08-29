#! /bin/python
import sys
import agent
import sim as simulation
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import numpy as np

import cProfile, pstats, io # Profiling stuff

def calc_reward(state, nextstate):
    return state[7]*2 - 1

def do_cycle(agent, return_observation=False):
    env = simulation.Simulation([-60, 0, 1000, 0, 0], dt=0.1)
    totalreward = 0

    action = 1
    observation, done = env.step(action)
    observations = [env.state]

    for _ in range(300):
        action = agent.get_action(observation)
        nextobservation, done = env.step(action)
        reward = calc_reward(observation, nextobservation)

        totalreward += reward

        if done:
            break

        agent.update(observation, action, reward, nextobservation)

        observation = nextobservation
        observations.append(env.state)

    if return_observation:
        return totalreward, observations
    else:
        return totalreward

def do_simulation(doplot=True):
    aagent = agent.SARSAAgent(0.9, 0.9, 0.2)

    rewards = []
    numepochs = 200
    
    for epoch in range(1,numepochs+1):
        print(f"Epoch {epoch} / {numepochs}")
        reward = do_cycle(aagent)
        rewards.append(reward)

    _, observations = do_cycle(aagent, True)

    plt.plot(range(1,numepochs+1), rewards)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    observations = np.array(observations)
    
    hdiffs = np.diff(observations[:,2], prepend=0.0)

    if doplot:
        ax.scatter(observations[:,0], observations[:,1], observations[:,2],\
                c=hdiffs)
        plt.show()

def do_profile():
    pr = cProfile.Profile()
    pr.enable()
    do_simulation(doplot=False)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def main():
    if len(sys.argv) > 1:
        mode =  sys.argv[1]
        if mode == "run":
            do_simulation()
        elif mode == "profile":
            do_profile()
        else:
            print(f"Did not understand \"{mode}\".")
            sys.exit(1)
    else:
        do_simulation()

if __name__ == "__main__":
    main()
