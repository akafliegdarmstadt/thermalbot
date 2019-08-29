#! /bin/python
import agent
import sim as simulation
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import numpy as np

def calc_reward(state, nextstate):
    return state[7]*2 - 1

def do_cycle(agent, return_observation=False):
    env = simulation.Simulation([0, 0, 1000, 0, 0], dt=0.1)
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

def main():
    aagent = agent.SARSAAgent()

    rewards = []
    numepochs = 100
    
    for epoch in range(1,numepochs+1):
        print(f"Epoch {epoch} / {numepochs}")
        reward = do_cycle(aagent)
        rewards.append(reward)

    _, observations = do_cycle(aagent, True)

    plt.plot(range(1,numepochs+1), rewards)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    observations = np.array(observations)
    
    hdiffs = np.diff(observations[:,2], prepend=0.0)*1000

    ax.scatter(observations[:,0], observations[:,1], observations[:,2], c=hdiffs, norm=colors.PowerNorm(1/2))
    plt.show()
if __name__ == "__main__":
    main()
