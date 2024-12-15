import numpy as np
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import torch
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import torch
import pandas as pd

from forecastpnn.utils.metrics import form_predictions
from forecastpnn.utils.pred_utils import create_multistep_forecasts_interval

plt.rcParams['font.family'] = "Times New Roman" #"cmr10"
plt.rcParams.update({"axes.labelsize" : "large"}) # 'font.size': 11, 
#plt.rcParams['font.serif'] = "Computer Modern"

models = ["Epinowcast", "RIVM", "NowcastPNN"]
colors = ['dodgerblue', 'black', 'crimson']

## Make plot over entire dataset for desired confidence level
def plot_entire_confints(dataset, model, n_samples = 200, levels = [0.5, 0.95], weeks = False, xlims = None, random_split = True, test_idcs = None, total = True, dow = False):
    plotloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    mat, y = next(iter(plotloader))
    if dow:
        mat, dow_val = mat
        mat, dow_val, y = mat.to("cpu"), dow_val.to("cpu"), y.to("cpu").numpy()
    else:
        mat, prev = mat
        prev = prev.to("cpu")
        mat, y = mat.to("cpu"), y.to("cpu").numpy()
    model.eval() # sets batch norm to eval so a single entry can be passed without issues of calculating mean and std.
    model.drop1.train() # keeps dropout layers active
    model.drop2.train()
    model = model.to("cpu")
    preds = np.zeros((y.shape[0], n_samples))
    for i in range(n_samples):
        #preds[:, i] = np.squeeze(model(mat, prev).sample().numpy()) if not dow else np.squeeze(model(mat, dow_val).sample().numpy())
        #preds[:, i] = np.add(model(mat).detach().numpy(), np.squeeze(prev)) if not dow else np.squeeze(model(mat, dow_val).sample().numpy())
        preds[:, i] = np.add(np.squeeze(model(mat).sample().numpy()), np.squeeze(prev.clone())) if not dow else np.squeeze(model(mat, dow_val).sample().numpy())
    preds_median = np.quantile(preds, 0.5, axis=1)

    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(y, label=r"True count", c = "black")
    plt.plot(prev, label=r"Previous observation", c = "dodgerblue")
    #plt.plot(y_atm, label="reported on day", c = "darkgrey")
    plt.plot(preds_median, label = r"One-Step Forecast predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = intervals_dict[l]
        plt.fill_between(range(len(y)), lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    if not random_split:
        if weeks:
            plt.axvline(300, color = "black", label=r"division train/test", linestyle="--")
        else:
            plt.axvline(2100, color = "black", label=r"division train/test", linestyle="--")
    """if test_idcs is not None:
        plt.vlines(test_idcs, ymin=0, ymax=1000, linewidth = 0.1, colors = "red", label = "Test set")"""
    plt.xlabel(fr"{'EpiWeeks' if weeks else 'Days'} since start of observation")
    plt.legend()
    plt.ylabel(r"Number of cases")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    if xlims is not None:
        plt.xlim(xlims)
        plt.savefig(fr".figures/nowcast_{'week' if weeks else 'day'}_subset_{xlims[0]}_{xlims[1]}")            
    elif not random_split:
        if weeks:
            plt.xlim(300, 418)
        else:
            plt.xlim(2133, 2844)
        plt.savefig(fr".figures/nowcast_{'week' if weeks else 'day'}_recent")
    else: 
        plt.savefig(fr".figures/nowcast_{'week' if weeks else 'day'}")
    plt.show()


def plot_confints_forecast(dataset, model, n_samples = 200, levels = [0.5, 0.95], idx = 2275, steps = 30, steps_before = 30, weeks = False):
    """Plot the confidence intervals for a single forecast.
    
    Args    
    ----
    `dataset` [torch.utils.data.Dataset]: The dataset to use for forecasting.
    `model` [torch.nn.Module]: The model to use for forecasting.
    `n_samples` [int]: The number of samples to use for the forecast.
    `levels` [list]: The confidence levels to use for the forecast.
    `idx` [int]: The index to use for the forecast.
    `steps` [int]: The number of steps to forecast.
    `steps_before` [int]: The number of steps before the forecast.
    """
    (forecasts, pred_median), y = create_multistep_forecasts_interval(dataset, idx, model, steps=steps, return_true=True)
    y_before = [dataset.__getitem__(i)[1].cpu().numpy() for i in range(idx-steps_before, idx)]

    plt.figure(figsize=(10, 6))
    plt.plot(range(idx-steps_before, idx), y_before, label=r"True count", c = "black")
    plt.plot(range(idx, idx+steps), y, label=r"True count", c = "black")
    plt.plot(range(idx, idx+steps), pred_median, label = r"Forecast predictions", c = "crimson", alpha = 0.75)
    for l in levels:
        lower, upper = forecasts[l]
        plt.fill_between(range(idx, idx+steps), lower, upper, color = "crimson", alpha = 0.2, label = f"{int(100*l)}% CI")
    plt.grid(alpha=.2)
    plt.xlabel(fr"{'EpiWeeks' if weeks else 'Days'}")
    plt.legend()
    plt.ylabel(r"Number of cases")
    plt.savefig(fr".figures/forecast_{idx}_{steps}s_{steps_before}b_{'week' if weeks else 'day'}")
    plt.show()