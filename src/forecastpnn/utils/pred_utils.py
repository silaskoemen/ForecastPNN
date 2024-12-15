import torch
import numpy as np
from tqdm import tqdm

def create_multistep_forecasts_point(dataset, idx, model, steps=7, return_true=False):
    """Function to create multi-step forecasts.

    Args
    ----
    `input_data` [torch.Tensor]: Input data to use for forecasting.
    `model` [torch.nn.Module]: Model to use for forecasting.
    `steps` [int]: Number of steps to forecast.

    Returns
    -------
    [torch.Tensor]: Multi-step forecasts.
    """
    forecasts = []
    if return_true:
        true_values = []
    model.eval()
    model.to("cpu")
    model.drop1.train()
    model.drop2.train()
    model.drop3.train()
    for i in range(steps):
        (mat, p), y = dataset.__getitem__(idx+i)
        if i > 0:
            mat[-1, 0] = torch.tensor(forecast/dataset.get_max_val())
        forecast = model(torch.unsqueeze(mat.to("cpu"), 0)).sample().numpy()
        forecasts.append(forecast)
        if return_true:
            true_values.append(y.cpu().numpy())
    if return_true:
        return np.squeeze(np.column_stack(forecasts)), np.squeeze(np.column_stack(true_values))
    return np.column_stack(forecasts)


def create_multistep_forecasts_interval(dataset, idx, model, steps = 7, n_samples = 2000, levels = [0.5, 0.95], return_true = False):
    """Function to create multi-step forecasts.

    Args
    ----
    `input_data` [torch.Tensor]: Input data to use for forecasting.
    `model` [torch.nn.Module]: Model to use for forecasting.
    `steps` [int]: Number of steps to forecast.

    Returns
    -------
    [torch.Tensor]: Multi-step forecasts.
    """
    if return_true:
        true_values = []
    model.eval() # sets batch norm to eval so a single entry can be passed without issues of calculating mean and std.
    model.drop1.train() # keeps dropout layers active
    model.drop2.train()
    model.drop3.train()
    model = model.to("cpu")
    preds = np.zeros((steps, n_samples))

    for s in tqdm(range(steps), "Generating samples per forecast step"):
        if s == 0:
            (mat, p), y = dataset.__getitem__(idx+s)
            mat, y = mat.to("cpu"), y.to("cpu").numpy()
            for i in range(n_samples):
                preds[s, i] = np.squeeze(model(torch.unsqueeze(mat, 0)).sample().numpy())
        else:
            (new_mat, p), y = dataset.__getitem__(idx+s)
            new_mat, y = new_mat.to("cpu"), y.to("cpu").numpy()
            new_mat[:-1, :] = mat[1:, :]
            new_mat[-1, 0] = torch.tensor(np.quantile(preds[s-1, :], 0.5)/dataset.get_max_val())
            for nq, q in enumerate([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]):
                for i in range(n_samples//10):
                    preds[s, i + nq*(n_samples//10)] = np.squeeze(model(torch.unsqueeze(new_mat, 0)).sample().numpy())

        if return_true:
            true_values.append(y)

    preds_median = np.quantile(preds, 0.5, axis=1)

    intervals_dict = {}
    for l in levels:
        intervals_dict[l] = (np.quantile(preds, (1-l)/2, 1), np.quantile(preds, (1+l)/2, 1))
    if return_true:
        return (intervals_dict, preds_median), np.squeeze(np.column_stack(true_values))
    else:
        return (intervals_dict, preds_median)