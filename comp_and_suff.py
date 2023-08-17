import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from synth_datasets import MyDataset
from utils import plot_saliency

model_type = "cnn"
dataset_name = "wafer"
exp_name = "2023-01-22 15_53 enable_season=False"
exp_name = ""
perturb_type = "opposite class mean"
perturb_type = "blur"

for saliency_name in ["ig", "deepshap", "fo", "afo", "shap", "lime"]:
# for saliency_name in ["str-saliency"]:
    print("\n"+"="*40)
    print("saliency_name:", saliency_name, ", dataset_name:", dataset_name, "model_type:", model_type)
    model: torch.nn.Module = torch.load("./models/trained_models/%s_%s_best"%(model_type, dataset_name))
    model.cuda()
    model.device = "cuda"
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
        
    saliency_map_dir = os.path.join("./generated_maps", model_type, dataset_name, saliency_name, exp_name)

    label_0_series = []
    label_1_series = []
    for i in range(len(dataset)):
        X, y = dataset[i]
        label = y.item()
        if label==0:
            label_0_series.append(X.flatten().cpu().detach().numpy())
        elif label==1:
            label_1_series.append(X.flatten().cpu().detach().numpy())
    if perturb_type=="blur":
        label_0_mean = label_1_mean = gaussian_filter(X.flatten().cpu().detach().numpy(), sigma=2)
    elif perturb_type=="opposite class mean":
        label_0_mean = np.mean(label_0_series, axis=0)
        label_1_mean = np.mean(label_1_series, axis=0)
    else:
        raise ValueError("Wrong perturbation type! ")
    print(label_0_mean.shape)

    all_instances_predictions_deletion = []
    all_instances_predictions_insertion = []
    for i in tqdm(range(len(dataset))):
        predictions_deletion = []
        predictions_insertion = []
        X, y = dataset[i]
        label = y.item()
        original_X = X.clone().flatten().cpu().detach().numpy()
        zero_reference_X = label_0_mean.copy() if label==1 else label_1_mean.copy()
        if saliency_name=="str-saliency":
            saliency_map_trend = np.load(os.path.join(saliency_map_dir, "%d_trend.npy"%(dataset.start_index+i))).flatten()
            saliency_map_remainder = np.load(os.path.join(saliency_map_dir, "%d_remainder.npy"%(dataset.start_index+i))).flatten()
            saliency_map = np.maximum(saliency_map_trend, saliency_map_remainder)
        else:
            saliency_map = np.load(os.path.join(saliency_map_dir, "%d.npy"%(dataset.start_index+i))).flatten()
        # plot_saliency(original_X, saliency_map)
        
        ######################## Evaluation metric ###############################
        
        top_steps = sorted(list(range(len(saliency_map))), key=lambda index: saliency_map[index], reverse=True) # Order of timesteps in saliency map
        original_prediction = model(X).item()
        zero_reference_prediction = model(torch.tensor(np.asarray([zero_reference_X]), dtype=torch.float32, device=model.device)).item()
        if label==0:
            original_prediction = 1-original_prediction
            zero_reference_prediction = 1-zero_reference_prediction
        predictions_deletion.append(original_prediction)
        predictions_insertion.append(zero_reference_prediction)
        # print("label: ", label)
        # print("zero_reference_prediction: ", zero_reference_prediction)
        for step in range(len(saliency_map)):
            index = top_steps[step]
            if label==0:
                original_X[index] = label_1_mean[index] # delete a timestep
                zero_reference_X[index] = X.flatten()[index] # insert a timestep
            elif label==1:
                original_X[index] = label_0_mean[index]
                zero_reference_X[index] = X.flatten()[index]
            new_prediction_deletion = model(torch.tensor(np.asarray([original_X]), dtype=torch.float32, device=model.device)).item()
            new_prediction_insertion = model(torch.tensor(np.asarray([zero_reference_X]), dtype=torch.float32, device=model.device)).item()
            if label==0:
                new_prediction_deletion = 1-new_prediction_deletion
                new_prediction_insertion = 1-new_prediction_insertion
            predictions_deletion.append(new_prediction_deletion)
            predictions_insertion.append(new_prediction_insertion)
            # print("new_prediction_deletion: ", new_prediction_deletion)
            # print("new_prediction_insertion: ", new_prediction_insertion)
            
        # plt.plot(predictions)
        # plt.show()
        # print("all deleted prediction: ", new_prediction_deletion)
        all_instances_predictions_deletion.append(predictions_deletion)
        all_instances_predictions_insertion.append(predictions_insertion)

    mean_predictions_deletion = np.mean(all_instances_predictions_deletion, axis=0)
    mean_predictions_insertion = np.mean(all_instances_predictions_insertion, axis=0)
    save_dir = "./CSresults/%s/%s/"%(model_type, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    # np.save("%s/deletion_%s.npy"%(save_dir, saliency_name), mean_predictions_deletion)
    # np.save("%s/insertion_%s.npy"%(save_dir, saliency_name), mean_predictions_insertion)
    print("Area under mean_predictions_deletion: ", np.sum(mean_predictions_deletion))
    print("Comprehensiveness: ", mean_predictions_deletion[0]-np.sum(mean_predictions_deletion)/len(mean_predictions_deletion))
    # plt.plot(mean_predictions_deletion)
    # plt.xlabel("Deleted steps")
    # plt.ylabel("Predicted probability of the original label")
    # plt.show()
    print("Area under mean_predictions_insertion: ", np.sum(mean_predictions_insertion))
    print("Sufficiency: ", mean_predictions_deletion[0]-np.sum(mean_predictions_insertion)/len(mean_predictions_insertion))
    # plt.plot(mean_predictions_insertion)
    # plt.xlabel("Inserted steps")
    # plt.ylabel("Predicted probability of the original label")
    # plt.show()