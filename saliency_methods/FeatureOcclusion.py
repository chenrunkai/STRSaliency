import numpy as np
import torch

from utils import plot_saliency


class FeatureOcclusion:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        pass
    
    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False):
        if data.dim()==3:
            data = data.squeeze(dim=0)
        saliency_map = []
        if detailed:
            print("Input shape:", data.shape)
        length = data.shape[-1]
        target = self.predict_fn(data)
        for i in range(length):
            mask = torch.ones([1, length]).cuda()
            mask[0][i] = 0
            new_input = data*mask
            new_output = self.predict_fn(new_input)
            loss: torch.Tensor = new_output-target
            if label==1:
                loss = -loss
            saliency_map.append(loss.item())
        saliency_map = np.asarray(saliency_map)
        if detailed:
            # print(saliency_map, len(saliency_map))
            plot_saliency(data, saliency_map)
        return saliency_map

class AugmentedFeatureOcclusion:
    def __init__(self, predict_fn, dataset, config: dict):
        self.dataset = dataset
        self.predict_fn = predict_fn
        self.config = config
    
    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False):
        np.random.seed(self.config["seed"])
        sample_times = self.config["sample_times"]
        if data.dim()==3:
            data = data.squeeze(dim=0)
        saliency_map = []
        if detailed:
            print("Input shape:", data.shape)
        length = data.shape[-1]
        target = self.predict_fn(data)
        for i in range(length):
            loss = 0
            new_input = data.clone()
            for sample_time in range(sample_times):
                sample_index = np.random.randint(0, len(self.dataset))
                X, y = self.dataset[sample_index]
                new_input[0][i] = X[0][i]
                new_output = self.predict_fn(new_input)
                loss += (new_output-target).item()
            if label==1:
                loss = -loss
            saliency_map.append(loss/sample_times)
        saliency_map = np.asarray(saliency_map)
        if detailed:
            # print(saliency_map, len(saliency_map))
            plot_saliency(data, saliency_map)
        return saliency_map