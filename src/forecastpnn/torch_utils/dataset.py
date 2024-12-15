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
        dow: bool = False,
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
        if dow:
            self.min_date = df.index.min()
        if isinstance(df, pd.DataFrame):
            self.df = np.array(df, dtype=np.float32)
        else:
            self.df = df
        self.max_val = np.max(self.df)
        self.past_units = past_units
        self.device = device
        self.dow = dow

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

        array = self.df[(idx - self.past_units):idx]
        target = self.df[idx]
        if self.dow:
            # Means daily data
            dow_val = self.idx_to_weekday(idx)
            dow_val = torch.tensor(dow_val).to(self.device)

        tensor = torch.from_numpy(array)
        tensor = tensor.to(device=self.device)
        target = torch.squeeze(torch.tensor([target]).to(self.device))

        if self.dow:
            return (tensor / self.max_val, dow_val), target
        return (tensor / self.max_val, tensor[-1]), target
