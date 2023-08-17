import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from saliency_methods.FeatureOcclusion import (AugmentedFeatureOcclusion,
                                               FeatureOcclusion)
from saliency_methods.LIMESaliency import LimeSaliency
from saliency_methods.SHAPSaliency import SHAP, DeepSHAP, IntegratedGradient
from saliency_methods.STRSaliency import STR, STRSaliency
from synth_datasets import MyDataset
from utils import plot_saliency

model_type = "cnn"
# model_type = "lstm"
# dataset_name = "cricket_x"
dataset_name = "remainder"
# saliency_name = "ig"
saliency_name = "str-saliency"
exp_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M ")+"enable_season=False"
# exp_name = ""

S = False
T = True
R = True
print("Seasonal: %s, Trend: %s, Remainder: %s"%(S, T, R) )


model: torch.nn.Module = torch.load("./models/trained_models/%s_%s_2"%(model_type, dataset_name))
# model: torch.nn.Module = torch.load("./models/trained_models/%s_%s_best"%(model_type, dataset_name))
# model = model.cuda()
model.eval()
if model_type=="lstm":
    # torch.backends.cudnn.enabled=False # Only RNN on CUDA has this problem. 
    model.train()
    # To calculate gradients for RNN on eval() mode, we need this line of code. 
    
if dataset_name=="wafer":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, int(1524*0.9), 1524) # for wafer
elif dataset_name=="ptb":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, int(1456*0.9), 1456) # for ptb
elif dataset_name=="cricket_x":
    dataset = MyDataset("./real_datasets/%s"%dataset_name, int(130*0.7), 130) # for cricket_x
else:
    dataset = MyDataset("./synth_datasets/%s"%dataset_name, 8000, 10000)
# dataloader = torch.utils.data.DataLoader(dataset)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if saliency_name == "str-saliency":
    saliency = STRSaliency(model, dataset.X, dataset.y, config["str-saliency"])
elif saliency_name=="fo":
    saliency = FeatureOcclusion(model)
elif saliency_name=="afo":
    saliency = AugmentedFeatureOcclusion(model, dataset, config["afo"])
elif saliency_name=="shap":
    saliency = SHAP(model, dataset.X, config["shap"])
elif saliency_name=="deepshap":
    saliency = DeepSHAP(model, dataset.X)
elif saliency_name=="ig":
    saliency = IntegratedGradient(model, dataset.X, config["ig"])
elif saliency_name=="lime":
    saliency = LimeSaliency(model, dataset.X, config["lime"])
else:
    raise ValueError("Wrong saliency name! ")

save_dir = os.path.join("./generated_maps", model_type, dataset_name, saliency_name, exp_name)
print("Saving to", save_dir)
if exp_name!="":
    os.makedirs(save_dir, exist_ok=False)

# print("Seed:", config["str-saliency"]["seed"])

for i in tqdm(range(0, len(dataset)), desc="Generating saliency map for %s, %s, %s"%(dataset_name, saliency_name, model_type)):
    X, y = dataset[i]
    if y.item()==0 and dataset_name in ["trend", "reremainder", "season"]:
        continue
    saliency_maps = saliency.generate_saliency(X, y, detailed=False, 
                                               enable_season=S, 
                                            #    enable_trend=T, 
                                            #    enable_remainder=R,
                                               )
    # plot_saliency(X.cpu().flatten().detach().numpy(), saliency_maps["remainder"])
    # raise
    if saliency_name=="str-saliency":
        for component in saliency_maps:
        # Save saliency map
            save_path = os.path.join(save_dir, str(i+dataset.start_index)+"_"+component)
            np.save(save_path, saliency_maps[component])
    else:
        saliency_map_wanted = saliency_maps
        # Save saliency map
        save_path = os.path.join(save_dir, str(i+dataset.start_index))
        np.save(save_path, saliency_map_wanted)
    # print("Saved to %s"%save_path)
    