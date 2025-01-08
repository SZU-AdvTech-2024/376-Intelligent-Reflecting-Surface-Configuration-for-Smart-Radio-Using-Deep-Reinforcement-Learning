
import numpy as np
import matplotlib.pyplot as plt


def plotOriginal():
    DDQN = np.load("outputs/DQN/.npy")
    DDQNAddDither = np.load("outputs/DQNAddDither/.npy")
    Random = np.load("outputs/Random/.npy")
    plt.plot(DDQN, color='r', label='DDQN')
    plt.plot(DDQNAddDither, color='b', label='DDQN+Dither')
    plt.plot(Random, color='c', label='Random')
    plt.xlim(0, 2000)
    plt.xlabel("Episode")
    plt.ylabel("performance $P_m$ (bps/Hz)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plotImproved():
    DDQN = np.load("outputs/DQN/.npy")
    DDQNAddDither = np.load("outputs/DQNAddDither/.npy")
    SAC = np.load("outputs/SAC/.npy")
    SACAddDither = np.load("outputs/SACAddDither/.npy")
    plt.plot(DDQN, color='r', label='DDQN')
    plt.plot(DDQNAddDither, color='b', label='DDQN+Dither')
    plt.plot(SAC, color='c', label='SAC')
    plt.plot(SACAddDither, color='m', label='SAC+Dither')
    plt.xlim(0, 2000)
    plt.xlabel("Episode")
    plt.ylabel("performance $P_m$ (bps/Hz)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plotOriginal()
    plotImproved()
