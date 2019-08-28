#! /bin/python
import agent
import sim as simulation
from matplotlib import pyplot as plt

def do_cycle(agent):
    env = simulation.Simulation([0, 0, 1000, 0, 0])
    totalreward = 0

    action = 1
    observation, reward, done = env.step(action)

    for _ in range(1000):
        action = agent.get_action(observation)
        nextobservation, reward, done = env.step(action)

        totalreward += reward

        if done:
            break

        agent.update(observation, action, reward, nextobservation)

    return totalreward

def main():
    randomagent = agent.RandomAgent()

    rewards = []
    numepochs = 100
    
    for epoch in range(1,numepochs+1):
        print(f"Epoch {epoch} / {numepochs}")
        reward = do_cycle(randomagent)
        rewards.append(reward)

    plt.plot(range(1,numepochs+1), rewards)
    plt.show()

if __name__ == "__main__":
    main()
