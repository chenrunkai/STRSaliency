
# Bad perturbation

from synth_datasets import MyDataset

# dataset = MyDataset("./synth_datasets/%s"%"remainder", 8000, 10000)


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def AUC(curve):
    return (np.sum(curve)-(curve[0]+curve[-1])/2)/len(curve-1)

print(plt.style.available)
plt.style.use(['ggplot'])
plt.rcParams['axes.facecolor'] = "e9ecef"

# For two curves

# saliency_names = ["ig", "deepshap", "fo", "afo", "lime", "shap", "str-saliency"]
# saliency_formal_names = ["IG", "DeepSHAP", "FO", "AFO", "LIME", "SHAP", "STR-Saliency"]


# colors_list = ["#611e1e", "#cd4432", "#ef9163", "#ecc29b", "#f3d9be"]
# colors_list = ["#F0EAD2", "#DDE5B6", "#ADC178", "#7B8F4B", "#A98467", "#6C584C", "#79675C"]
# for i in range(len(saliency_names)):
#     saliency_name = saliency_names[i]
#     curve = np.load("./CSresults/cnn/ptb/deletion_%s.npy"%(saliency_name))
#     print("AUC deletion of %s:"%saliency_name, AUC(curve))
#     chosen_indexes = np.linspace(0, len(curve)-1, 11, dtype=int)
#     # print(chosen_indexes)
#     plt.plot(np.linspace(0, 100, 11), curve[chosen_indexes], 
#              label=saliency_formal_names[i], 
#              linewidth=2, linestyle="-", 
#              marker="D", markersize=5, 
#              color = None if i!=6 else "#3a015c"
#              )
# plt.xlabel("Deleted steps (%)")
# plt.ylabel("Predicted probability of the original label")
# plt.legend()
# plt.show()

# for i in range(len(saliency_names)):
#     saliency_name = saliency_names[i]
#     curve = np.load("./CSresults/cnn/ptb/insertion_%s.npy"%(saliency_name))
#     print("AUC insertion of %s:"%saliency_name, AUC(curve))
#     chosen_indexes = np.linspace(0, len(curve)-1, 11, dtype=int)
#     plt.plot(np.linspace(0, 100, 11), curve[chosen_indexes], 
#              label=saliency_formal_names[i], 
#              linewidth=2, linestyle="-", 
#              marker="D", markersize=5, 
#              color = None if i!=6 else "#3a015c"
#              )
# plt.xlabel("Inserted steps (%)")
# plt.ylabel("Predicted probability of the original label")
# plt.legend()
# plt.show()


# For example illustration

# X, y = dataset[2]
# X = X.flatten().cpu().detach().numpy()
# X, y
# colors_list = ["#611e1e", "#cd4432", "#ef9163", "#ecc29b", "#f3d9be"]
# cmap = ListedColormap(colors_list, name = 'mycmap')

plt.figure(figsize=(5, 4))
series = np.random.normal(loc=1,scale=0.03,size=100)+np.sin(np.arange(100)/2)/30+(np.arange(100)-50)**3/200000-(np.arange(100)-50)/150
plt.ylim(0, 1.5)
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.plot(series, linewidth=2, color="#cd4432")
plt.show()


plt.figure(figsize=(5, 4))
series[45:50] = series[45:50]*0.5
plt.ylim(0, 1.5)
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.plot(series, linewidth=2, color="#cd4432")
# rect1 = mpatches.Rectangle((43.5, 0), 2, 100, facecolor="#f3d9be")
# rect2 = mpatches.Rectangle((48.5, 0), 2, 100, facecolor="#f3d9be")
# plt.gca().add_patch(rect1)
# plt.gca().add_patch(rect2)
rect = mpatches.Rectangle((43.5, 0), 7, 100, facecolor="#f3d9be")
plt.gca().add_patch(rect)
plt.show()


