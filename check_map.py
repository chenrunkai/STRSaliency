import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from saliency_methods.STRSaliency import STR
from synth_datasets import MyDataset
from utils import plot_saliency

model_type = "cnn"
dataset_name = "remainder"
saliency_name = "str-saliency"
exp_time = "2023-01-18 19_10 remainder+trend"
print("saliency_name: ", saliency_name, "dataset_name: ", dataset_name, "model_type: ", model_type)
dataset = MyDataset("./synth_datasets/%s"%dataset_name, 8000, 10000)
# dataloader = torch.utils.data.DataLoader(dataset)


saliency_map_dir = os.path.join("./generated_maps", model_type, dataset_name, saliency_name, exp_time)

# print("Seed:", config["str-saliency"]["seed"])

total_attr_sums = []
spikes_attr_sums = []
spike_areas= []
precisions = []
recalls = []

for i in range(len(dataset)//10):
    X, y = dataset[i]
    if y.item()==0:
        continue
    saliency_map = np.load(os.path.join(saliency_map_dir, "%d_remainder.npy"%(dataset.start_index+i))).flatten()
    # plot_saliency(X.flatten().cpu().detach().numpy(), saliency_map)
    
    ######################## Seasonal ########################################
    
    # if np.max(saliency_map)>0:
    #     saliency_map /= np.max(saliency_map) # Norm so that max attribution is 1
    # else:
    #     print("All zero! ")
    # total_attr_sums.append(np.sum(saliency_map))
    # spikes_attr_sums.append(saliency_map[10])
    # spike_areas.append(1)
    # recalls.append(saliency_map[10]/1)
    
    # continue
        
    ######################## Evaluation metric ###############################
    
    # Search for spikes
    _, _, _, remainder = STR(X.flatten().detach().cpu().numpy(), plot=False)
    spikes = []
    for i in range(len(remainder)):
        if remainder[i]>=0.2:
            spikes.append(i)
    # print(spikes)
    
    # Calculate precision and recall
    saliency_map = np.maximum(saliency_map, 0)
    if np.max(saliency_map)>0:
        saliency_map /= np.max(saliency_map) # Norm so that max attribution is 1
    else:
        # raise ValueError("There is a map that is all 0! Index=%d, map=%s"%(i, saliency_map))
        pass
    
    # Plot map
    # plt.figure(figsize=(4, 3))
    # plot_saliency(X, saliency_map)
    # raise
    
    
    total_attr_sum = np.sum(saliency_map)
    spikes_attr_sum = 0
    for i in spikes:
        spikes_attr_sum += saliency_map[i]
    total_attr_sums.append(total_attr_sum)
    spikes_attr_sums.append(spikes_attr_sum)
    spike_areas.append(len(spikes))
    if total_attr_sum>0:
        precisions.append(spikes_attr_sum/total_attr_sum)
    recalls.append(spikes_attr_sum/len(spikes))

print("%s | Sum attribution: %f | Spikes attribution: %f | Spikes: %f | Precision: %f | Recall: %f"
      %(saliency_name, sum(total_attr_sums), sum(spikes_attr_sums), sum(spike_areas), 
        sum(spikes_attr_sums)/sum(total_attr_sums), sum(spikes_attr_sums)/sum(spike_areas))
      )

print("%s | Average Precision: %f | Average Recall: %f"
      %(saliency_name, np.mean(precisions), np.mean(recalls))
      )

# print("%s | Total Precision: %f | Average Recall: %f"
#       %(saliency_name, np.sum(spikes_attr_sums)/np.sum(total_attr_sums), np.sum(spikes_attr_sums)/np.sum(spike_areas))
#       )