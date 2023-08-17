import numpy as np
import shap
import torch
from tqdm import tqdm

from utils import plot_saliency


class SHAPHelper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        data = torch.tensor(data, dtype=torch.float32).cuda()
        result: torch.Tensor = self.model(data)
        result = result.cpu().detach().numpy()
        return result

class SHAP:
    def __init__(self, predict_fn, background_data, config: dict):
        self.config = config
        background_data = np.asarray(background_data)
        bg_summary = shap.kmeans(background_data, self.config["kmeans"])
        self.shap_explainer = shap.KernelExplainer(SHAPHelper(predict_fn), bg_summary, seed=self.config["seed"])
    
    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False):
        nsamples = self.config["nsamples"]
        shap_values = self.shap_explainer.shap_values(data.cpu().detach().numpy(), nsamples=self.config["nsamples"])[0]
        if label==0:
            shap_values = -shap_values
        if np.max(np.abs(shap_values))>0:
            shap_values /= np.max(np.abs(shap_values)) # Normalize
        if detailed:
            print(shap_values)
            plot_saliency(data, shap_values)
        return shap_values

class DeepSHAP:
    def __init__(self, predict_fn, background_data):
        background_data = torch.as_tensor(np.asarray(background_data), dtype=torch.float32)
        # bg_summary = shap.kmeans(background_data, 50)
        predict_fn.cpu()
        predict_fn.device = "cpu"
        self.shap_explainer = shap.DeepExplainer(predict_fn, background_data)
    
    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False):
        shap_values = self.shap_explainer.shap_values(data)[0]
        if label==0:
            shap_values = -shap_values
        if np.max(np.abs(shap_values))>0:
            shap_values /= np.max(np.abs(shap_values)) # Normalize
        if detailed:
            print(shap_values)
            plot_saliency(data, shap_values)
        return shap_values
    
class IntegratedGradient:
    def __init__(self, predict_fn, background_data, config: dict):
        background_data = torch.as_tensor(np.asarray([[x] for x in background_data]), dtype=torch.float32).cuda()
        self.shap_explainer = shap.GradientExplainer(predict_fn, background_data)
        self.config = config
    
    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False):
        shap_values = self.shap_explainer.shap_values(data.unsqueeze(0), rseed=self.config["seed"])[0]
        if label==0:
            shap_values = -shap_values
        if np.max(np.abs(shap_values))>0:
            shap_values /= np.max(np.abs(shap_values)) # Normalize
        if detailed:
            print(shap_values)
            plot_saliency(data, shap_values)
        return shap_values