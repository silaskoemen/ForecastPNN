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
        The dataset will generate matrices for each date within this range.

        Parameters:
        - start_date: The start date for generating matrices.
        - end_date: The end date for generating matrices.
        - past_days: Number of past days to consider for each matrix.
        - max_delay: Maximum delay to consider for each matrix.
        """
        if dow:
            self.min_date = self.df.index.min()
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
        # Calculate the number of days between 60 days after start_date and 46 days before end_date
        return len(self.df) - (self.past_units - 1)

    def __getitem__(self, idx):
        # Calculate the date for the current iteration, considering the adjusted range
        idx += self.past_units
        assert idx < self.df.shape[0], "Index out of range"

        array = self.df[(idx - self.past_units):idx]
        target = self.df[idx]
        if self.dow:
            # Means daily data
            dow_val = self.idx_to_weekday(idx)
            dow_val = torch.tensor(dow_val).to(self.device)

        tensor = torch.from_numpy(array)
        tensor = tensor.to(device=self.device)
        target = torch.tensor([target]).to(self.device)

        if self.dow:
            return (tensor / self.max_val, dow_val), target
        return tensor / self.max_val, target
