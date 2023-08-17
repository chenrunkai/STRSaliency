import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from saliency_methods.STRSaliency import STR

index = np.linspace(0, 100, 100, endpoint=False)

def synth():
    season_x = np.random.randint(0, 2)
    trend_x = np.random.randint(0, 2)
    remainder_x = np.random.randint(0, 2)
    season = np.zeros(100)
    if np.random.randint(0, 2)==1:
        season += np.sin(index*np.pi/10+np.random.random()*2*np.pi)/5
    else:
        season += np.sin(index*np.pi/12.5+np.random.random()*2*np.pi)/5
    if season_x==1:
        season += np.sin(index*np.pi/5+np.random.random()*2*np.pi)/5
        
    trend = ((index-50)/50)**3-(index-50)/50 if trend_x==0 else -((index-50)/50)**3+(index-50)/50
    
    remainder = np.random.normal(0, 0.05, 100)
    spikes = []
    if remainder_x==1:
        spike_num = np.random.randint(1, 4)
        for j in range(spike_num):
            i = np.random.randint(1, 99)
            if i not in spikes:
                remainder[i] += 0.5
                spikes.append(i)
            
    print("season_x:", season_x, "trend_x:", trend_x, "remainder_x:", remainder_x, "spikes:", spikes)
    
    return season+trend+remainder, (season_x, trend_x, remainder_x), spikes

for i in range(10000):
    data, (x_s, x_t, x_r), spikes = synth()
    label = x_t
    np.save("./synth_datasets/trend/data_%d.npy"%i, data)
    np.save("./synth_datasets/trend/label_%d.npy"%i, label)
    np.save("./synth_datasets/trend/spikes_%d.npy"%i, spikes)
    # plt.plot(data)
    # plt.show()
    # STR(data, True)