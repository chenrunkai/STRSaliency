import lime
import numpy as np
import torch

from utils import plot_saliency


class LimeSaliency:
    class LimeHelper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x = torch.tensor(x, dtype=torch.float32, device=self.model.device)
            result: np.ndarray = self.model(x).cpu().detach().numpy()
            result = np.concatenate((result, 1-result), axis=1)
            return result

    def __init__(self, predict_fn, background_data, config: dict):
        self.config = config
        self.predict_fn = predict_fn
        self.wrapped_predict_fn = self.LimeHelper(self.predict_fn)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            np.asarray(background_data), verbose=False, random_state=self.config["seed"])

    def generate_saliency(self, data, label, detailed: bool = False):
        if isinstance(data,  torch.Tensor):
            data = data.flatten().cpu().detach().numpy()
        timesteps = data.shape[0]
        lime_gbr = self.lime_explainer.explain_instance(
            data, self.wrapped_predict_fn, num_features=timesteps, 
            num_samples = self.config["num_samples"])
        lime_values = np.zeros(timesteps)
        for ids, val in lime_gbr.local_exp[1]:  # np.argmax(fetch_class(d))
            lime_values[ids] = val
        if np.max(np.abs(lime_values))>0:
            lime_values /= np.max(np.abs(lime_values))
        if detailed:
            print(lime_gbr)
            print(lime_values)
            plot_saliency(data, lime_values)
        return lime_values