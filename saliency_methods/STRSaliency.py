import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from utils import plot_saliency


def STR(input_: np.ndarray, enable_season=True, enable_trend=True, enable_remainder=True, plot=False):
    '''
    Returns:
    input, season(in fft), trend, remainder
    '''
    input_ = input_.flatten()
    season_fft: np.ndarray = np.fft.rfft(input_)
    for i in range(len(season_fft)):
        if i<=2:
            season_fft[i] = 0
    order = sorted(list(range(len(season_fft))), key=lambda x: np.absolute(season_fft[x]), reverse=True)
    # max_abs = np.max(np.absolute(fft_input))
    threshold = np.absolute(season_fft[order[2]])
    # threshold = np.absolute(season_fft[order[0]])
    if not enable_season:
        threshold = 99999999
    for i in range(len(season_fft)):
        if np.absolute(season_fft[i])<threshold:
            season_fft[i] = 0
    recovered_season = np.fft.irfft(season_fft, n=len(input_))
    trend: np.ndarray = input_-recovered_season
    if not enable_trend:
        trend = np.zeros(trend.shape)
    elif not enable_remainder:
        pass
    else:
        trend: np.ndarray = gaussian_filter(input_-recovered_season, sigma=2)
    remainder = input_-recovered_season-trend
    if plot:
        plt.style.use(['ggplot'])
        plt.rcParams['axes.facecolor'] = "e9ecef"
        plt.subplot(311)
        plt.plot(input_, label="Original", color="#457b9d")
        plt.plot(trend, label="Trend", color="#d4a373")
        plt.legend()
        plt.title("Original input and trend", fontsize=10)
        plt.subplot(312)
        plt.plot(remainder, color="#d4a373")
        plt.title("Remainder", fontsize=10)
        plt.subplot(313)
        plt.plot(np.absolute(season_fft), color="#d4a373")
        # plt.hlines(threshold, 0, len(fft_input), colors=["orange"])
        plt.title("Seasonal in frequency domain (absolute value)", fontsize=10)
        plt.tight_layout()
        plt.show()

    return input_, season_fft, trend, remainder


class STRSaliency:
    def __init__(self, predict_fn, background_data, background_label, config: dict):
        self.background_data = background_data
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        if "lr" not in self.config:
            self.config["lr"] = 1e-3
        if "max_iter" not in self.config:
            self.config["max_iter"] = 10000
        if "lambda_pred" not in self.config:
            self.config["lambda_pred"] = 1000
            print("lambda_pred is set to %d"%self.config["lambda_pred"])
        if "lambda_smoothness_r" not in self.config:
            self.config["lambda_smoothness_r"] = 1.0
        if "lambda_smoothness_t" not in self.config:
            self.config["lambda_smoothness_t"] = 10.0
        if "lambda_exceed" not in self.config:
            self.config["lambda_exceed"] = 1e3
        if "lambda_remainder" not in self.config:
            self.config["lambda_remainder"] = 1.0
        if "lambda_trend" not in self.config:
            self.config["lambda_trend"] = 1.0
        if "lambda_season" not in self.config:
            self.config["lambda_season"] = 5.0
        if "total_budget" not in self.config:
            self.config["total_budget"] = 0.3
        if "perturbation" not in self.config:
            self.config["perturbation"] = True
        if "seed" not in self.config:
            self.config["seed"] = None
        # if "early_stopping_threshold" not in self.config:
        #     self.config["early_stopping_threshold"] = 1e-3
        self.cross_entropy = torch.nn.BCELoss().to(self.device)
        self.predict_fn.to(self.device)

    def generate_saliency(self, data: torch.Tensor, label: int, detailed: bool = False,
                          enable_season=True, enable_trend=True, enable_remainder=True,
                          use_sample_as_perturbation=False):
        # Set up all seeds. 
        seed = self.config["seed"]
        if seed is not None and seed!=False:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            
        # data = data.reshape([1, -1])
        with torch.no_grad():
            target = self.predict_fn(data)
        [input_, season_fft, trend, remainder] = STR(data.cpu().detach().numpy(),
                                                     enable_season=enable_season, 
                                                     enable_trend=enable_trend, 
                                                     enable_remainder=enable_remainder,
                                                     plot=detailed)
        length = trend.shape[0]
        # [input_, season_fft, trend, remainder] = [
        #     data.flatten().cpu().detach().numpy(), 
        #     np.zeros(season_fft.shape), 
        #     np.zeros(length), 
        #     data.flatten().cpu().detach().numpy(), 
        # ] # For ablation study
        
        trend = torch.tensor(trend, dtype=torch.float32, device=self.device)
        season_fft = torch.tensor(season_fft, dtype=torch.complex32, device=self.device)
        remainder = torch.tensor(remainder, dtype=torch.float32, device=self.device)
        saliency_t = torch.normal(mean=0.8, std=0.1, size=trend.size(), dtype=torch.float32, device=self.device, requires_grad=True)
        saliency_s = torch.normal(mean=0.8, std=0.1, size=season_fft.size(), dtype=torch.float32, device=self.device, requires_grad=True)
        saliency_r = torch.normal(mean=0.8, std=0.1, size=remainder.size(), dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([saliency_t, saliency_s, saliency_r], lr=self.config["lr"])
        lambda_pred = self.config["lambda_pred"]
        lambda_smoothness_r = self.config["lambda_smoothness_r"]
        lambda_smoothness_t = self.config["lambda_smoothness_t"]
        lambda_trend = self.config["lambda_trend"]
        lambda_season = self.config["lambda_season"]
        lambda_remainder = self.config["lambda_remainder"]
        total_budget = self.config["total_budget"]

        minimum_loss = 1e9 # infinity

        bg_trend = torch.zeros(size=trend.size(), dtype=torch.float32, device=self.device)
        bg_season_fft = torch.zeros(size=season_fft.size(), dtype=torch.float32, device=self.device)
        cnt = 0
        for i in range(len(self.background_data)):
            sample = self.background_data[i]
            sample_label = self.background_label[i]
            if sample_label==label: # Want the opposite class's background
                continue # do not continue if the second perturbation is selected
            [_, s, t, _] = STR(sample)
            bg_trend += torch.tensor(t, dtype=torch.float32, device=self.device)
            bg_season_fft += torch.tensor(np.absolute(s), dtype=torch.float32, device=self.device) # Use absolute value
            cnt += 1
        bg_trend /= cnt
        bg_season_fft /= cnt
        # plt.plot(bg_trend.cpu().detach().numpy())
        # plt.title("Background Trend(Opposite class samples count: %d)"%cnt)
        # plt.show()
        
        # If perturbation is based on sampling: 
        bg_trend_samples = []
        bg_season_fft_samples = []
        if use_sample_as_perturbation:
            for sample_id in np.random.randint(0, len(self.background_data), size=20):
                sample = self.background_data[sample_id]
                [_, s, t, _] = STR(sample)
                bg_trend_samples.append(torch.tensor(t, dtype=torch.float32, device=self.device))
                bg_season_fft_samples.append(torch.tensor(np.absolute(s), dtype=torch.float32, device=self.device)) # Use absolute value
            pass
        
        stop_cnt = 0
        for i in range(self.config["max_iter"]):            
            if use_sample_as_perturbation:
                loss_pred = torch.zeros(target.size)
                for sample_id in range(len(bg_trend_samples)):
                    if self.config["perturbation"]:
                        perturbed_data: torch.Tensor = (1-saliency_t)*trend + saliency_t*bg_trend_samples[sample_id] \
                                    + torch.fft.irfft((1-saliency_s)*season_fft + saliency_s*bg_season_fft_samples[sample_id], n=length) \
                                    + (1-saliency_r)*remainder
                    else:
                        perturbed_data: torch.Tensor = saliency_t*trend + (1-saliency_t)*bg_trend_samples[sample_id] \
                                    + torch.fft.irfft(saliency_s*season_fft + (1-saliency_s)*bg_season_fft_samples[sample_id], n=length) \
                                    + saliency_r*remainder
                    perturbed_data.unsqueeze_(dim=0).unsqueeze_(dim=0)
                    prediction: torch.Tensor = self.predict_fn(perturbed_data)
                    loss_pred += target-prediction
            else:
                if self.config["perturbation"]:
                    perturbed_data: torch.Tensor = (1-saliency_t)*trend + saliency_t*bg_trend \
                                + torch.fft.irfft((1-saliency_s)*season_fft + saliency_s*bg_season_fft, n=length) \
                                + (1-saliency_r)*remainder
                else:
                    perturbed_data: torch.Tensor = saliency_t*trend + (1-saliency_t)*bg_trend \
                                + torch.fft.irfft(saliency_s*season_fft + (1-saliency_s)*bg_season_fft, n=length) \
                                + saliency_r*remainder
                perturbed_data.unsqueeze_(dim=0).unsqueeze_(dim=0)
                prediction: torch.Tensor = self.predict_fn(perturbed_data)
                loss_pred = target-prediction # New
                # print("label: %d\ntarget: %.3f\npred: %.3f\nloss_pred: %.3f"%(label,target,prediction,loss_pred))
                # print(target)
                # input()
                
            
            if label==0:
                loss_pred = -loss_pred
            if self.config["perturbation"]:
                loss_pred = -loss_pred # The goal is to make the model predict badly
            
            loss_budget_t = torch.norm(saliency_t, p=1)/length
            loss_budget_r = torch.norm(saliency_r, p=1)/length # L1-Norm of the saliency map
            loss_budget_s = torch.norm(saliency_s, p=1)/len(season_fft)
            loss_budget = loss_budget_t*lambda_trend+loss_budget_s*lambda_season+loss_budget_r*lambda_remainder
            loss_smoothness = torch.norm(saliency_r[:-1]-saliency_r[1:], p=1)/length*lambda_smoothness_r \
                            + torch.norm(saliency_t[:-1]-saliency_t[1:], p=1)/length*lambda_smoothness_t
            loss: torch.Tensor = loss_pred*lambda_pred \
                               + torch.relu(loss_budget-total_budget) \
                               + loss_smoothness
            loss.requires_grad_(True)
            optimizer.zero_grad()
            # print("label: %d\ntarget: %.3f\npred: %.3f\nloss_pred: %.3f\nloss_budget: %.3f\nloss_smothness: %.3f\nloss: %.3f"%(label,target,prediction,loss_pred,loss_budget,loss_smoothness,loss))
            # input()
            loss.backward()
            optimizer.step()
            # Clamp the maps to (0, 1)
            saliency_s.data.clamp_(0, 1)
            saliency_t.data.clamp_(0, 1)
            saliency_r.data.clamp_(0, 1)

            if (i+1)%(self.config["max_iter"]/50)==0:
                # if (i+1)%(self.config["max_iter"])==0:
                if detailed:
                    print("prediction: ", prediction, "target: ", target)
                    print("Iter: %d, loss_pred: %f, loss_budget: %f, loss_smoothness: %f, total: %f"
                        %(i, loss_pred, loss_budget, 
                        loss_smoothness, 
                        loss))
                if loss>minimum_loss-0.0001:
                    stop_cnt += 1
                else:
                    stop_cnt = 0
                if stop_cnt>=3:
                    break # early stopping
                minimum_loss = min(minimum_loss, loss)
                  
        if detailed or 1:
            print("prediction: ", prediction, "target: ", target)
            print("Iter: %d, loss_pred: %f, loss_budget: %f, loss_smoothness: %f, total: %f"
                    %(i, loss_pred, loss_budget, 
                        loss_smoothness, 
                        loss))
            plt.subplot(511)
            plt.plot(input_.flatten())
            plt.title("input")
            plt.subplot(512)
            plt.plot(perturbed_data.flatten().cpu().detach().numpy())
            plt.title("perturbed")
            plt.subplot(513)
            plt.plot(saliency_s.flatten().cpu().detach().numpy())
            plt.title("saliency_season")
            plt.subplot(514)
            plt.plot(saliency_t.flatten().cpu().detach().numpy())
            plt.title("saliency_trend")
            plt.subplot(515)
            plt.plot(saliency_r.flatten().cpu().detach().numpy())
            plt.title("saliency_remainder")
            plt.show()

            plot_saliency(trend, saliency_t)
            plot_saliency(torch.absolute(season_fft), saliency_s, isfreq=True)
            plot_saliency(remainder, saliency_r)
            # plot_saliency(data, saliency_t+saliency_r)

        saliency = {
            "trend": saliency_t.cpu().detach().numpy(),
            "season": saliency_s.cpu().detach().numpy(),
            "remainder": saliency_r.cpu().detach().numpy()
        }
        return saliency
