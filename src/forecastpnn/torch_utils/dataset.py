from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def days_to_date(start_date, num_days, past_units = 1):
    """
    Converts number of days since start_date to the corresponding date.
    
    Args:
    `start_date` [str]: The start date in 'YYYY-MM-DD' format.
    `num_days` [int]: Number of days from the start date.
    
    Returns:
    [datetime]: The corresponding date.
    """
    if not isinstance(start_date, datetime):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    return start_date + timedelta(days=int(num_days+past_units-1))

class ReportingDataset(Dataset):
    ## Theoretically, should contain covariates for date too, return tuple of matrix and covariates as well as label at each iteration

    def __init__(
        self,
        df: pd.DataFrame | np.ndarray,
        past_units: int = 12,
        time_features: bool = False,
        device: str = "mps",
    ):
        """
        Initialize the dataset with a start and end date.
        The dataset will generate the array for each date within this range.

        Parameters:
        `df` [pd.DataFrame | np.ndarray]: The data to be used for training.
        `past_units` [int]: The number of past units to consider for each prediction.
        `dow` [bool]: Whether to include day of the week as a feature.
        `device` [str]: The device to use for the tensor.
        """
        """ if dow:
            self.min_date = df.index.min() """
        if isinstance(df, pd.DataFrame):
            self.df = np.array(df, dtype=np.float32)
        else:
            self.df = df
        self.max_val = np.max(self.df[:, 0])
        self.past_units = past_units
        self.device = device
        self.time_features = time_features
        #self.dow = dow

    def get_length(self):
        return self.df.shape[0]
    
    def idx_to_weekday(self, idx):
        return days_to_date(start_date=self.min_date, num_days=idx, past_units=self.past_units).weekday()
    
    def get_max_val(self):
        return self.max_val
    
    def setmax_val(self, max_val):
        self.max_val = max_val

    def __len__(self):
        return self.df.shape[0] - self.past_units

    def __getitem__(self, idx):
        # Calculate the date for the current iteration, considering the adjusted range
        idx += self.past_units
        assert idx < self.df.shape[0], f"Index {idx} out of range {self.df.shape[0]}"

        array = self.df[(idx - self.past_units):idx, :]
        target = self.df[idx, 0]
        tensor = torch.from_numpy(array)
        tensor = tensor.to(device=self.device)
        prev = torch.tensor(0., dtype=torch.float32)#.0# tensor[-1, 0].clone()
        tensor[:, 0] /= self.max_val
        label = torch.squeeze(torch.tensor([target]).to(self.device))

        return (tensor, prev), label

class PercentageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame | np.ndarray,
        past_units: int = 42,
        device: str = "mps",
    ):
        if isinstance(df, pd.DataFrame):
            # Ensure columns are in the correct order for [count, perc_count]
            df = df[['count', 'perc_count']].to_numpy(dtype=np.float32)
        self.df = df
        self.past_units = past_units
        self.device = device

    def __len__(self):
        return self.df.shape[0] - self.past_units
    
    def __getitem__(self, idx):
        idx += self.past_units
        y = self.df[idx, 1]  # target percentage at current index
        past_perc_counts = self.df[idx - self.past_units:idx, 1]
        prev_true_count = self.df[idx - 1, 0]
        x_tensor = torch.from_numpy(past_perc_counts).unsqueeze(-1).to(self.device)
        prev_count_tensor = torch.tensor(prev_true_count, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        return (x_tensor, prev_count_tensor), y_tensor


class PercentageDatasetMultistep(Dataset):
    def __init__(
        self,
        df: pd.DataFrame | np.ndarray,
        past_units: int = 42,
        steps_ahead: int = 14,
        device: str = "mps",
    ):
        if isinstance(df, pd.DataFrame):
            df = df[['count', 'perc_count']].to_numpy(dtype=np.float32)
        self.df = df
        self.past_units = past_units
        self.steps_ahead = steps_ahead
        self.device = device

    def __len__(self):
        return self.df.shape[0] - self.past_units - (self.steps_ahead - 1)

    def __getitem__(self, idx):
        idx += self.past_units
        # Get future percentage values for next steps_ahead days
        y = self.df[idx:idx + self.steps_ahead, 1]
        # Get past percentage values
        past_perc_counts = self.df[idx - self.past_units:idx, 1]
        prev_true_count = self.df[idx - 1, 0]
        
        x_tensor = torch.from_numpy(past_perc_counts).unsqueeze(-1).to(self.device)
        prev_count_tensor = torch.tensor(prev_true_count, dtype=torch.float32, device=self.device)
        y_tensor = torch.from_numpy(y).to(self.device)
        
        return (x_tensor, prev_count_tensor), y_tensor