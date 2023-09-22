import numpy as np
import matplotlib.pyplot as plt


def plot_sumrwdperepi(sum_rewards):
    "trace courbe de somme des rec par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(sum_rewards)), sum_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
    
def plot_sumrwd_mean_perepi(sum_rewards,avgs):
    "trace courbe de somme des rec et moyenne glissante par episodes"
    print("sum_rwd:",type(sum_rewards))
    print("avgs:",type(avgs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(sum_rewards)), sum_rewards, label='sum_rwd')
    plt.plot(np.arange(len(avgs)), avgs, c='r', label='average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left');
    plt.show()
    
    

    
