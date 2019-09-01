#! /bin/python
import sys
import agent
import cProfile, pstats, io # Profiling stuff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from environment import Environment, thermal


def calc_reward(state, nextstate):
    return state[7]*2 - 1

def do_cycle(agent, return_observation=False):
    env = Environment([-60, 0, 1000, 10, 0, 0, 0.0])
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
    aagent = agent.SARSAAgent(0.1, 0.9, 0.02, 0.2)


    rewards = []
    numepochs = 10000
    
    for epoch in range(1,numepochs+1):
        
        reward = do_cycle(aagent)
        rewards.append(reward)

        print(f"Epoch {epoch} / {numepochs} - reward: {reward}")

    _, observations = do_cycle(aagent, True)

    plt.plot(range(1,numepochs+1), rewards)


    observations = np.array(observations)

    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(observations[:,0], observations[:,1], observations[:,2])

        dist = max(np.abs([*ax.get_xlim(), *ax.get_ylim()]))

        pts = np.linspace(-dist, dist)
        x_mg, y_mg = np.meshgrid(pts, pts)

        w = np.vectorize(thermal)(x_mg, y_mg, 1000)

        zlim = ax.get_zlim()
        ax.contour(x_mg, y_mg, w, zdir='z', offset=1000)

        ax.set_zlim(*zlim)

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

def main(do_plot=True):
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
        do_simulation(do_plot)

if __name__ == "__main__":
    main()
