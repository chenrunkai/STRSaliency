import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap


def plot_saliency(data, saliency, isfreq=False, title=""): # Custom
    CMAP = get_cmap("Greens")
    # print(data, saliency)
    data = data.flatten()
    saliency = saliency.flatten()
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.cpu().detach().numpy()
    plt.style.use("default")
    plt.figure(figsize=(4, 2))
    plt.clf()
    fig = plt.gcf()
    im = plt.imshow(saliency.reshape([1, -1]), cmap=CMAP, aspect="auto", alpha=0.85,
                    extent=[-0.5, len(saliency) - 1+0.5, float(np.min([np.min(data)])) - 1e-1,
                            float(np.max([np.max(data)])) + 1e-1]
                    )
    plt.plot(data, color="orange")
    plt.grid(False)
    plt.xlabel("Frequency" if isfreq else "Timestep")
    plt.ylabel("Value")
    plt.title(title)
    # cax = fig.add_axes([0.30, 0.1, 0.5, 0.03])
    plt.clim(0, 1) # Limit the range of color
    # fig.colorbar(im, cax=cax, orientation="horizontal", label="Saliency value")
    plt.tight_layout(pad=0)
    plt.show()
    pass

def setup_torch_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True