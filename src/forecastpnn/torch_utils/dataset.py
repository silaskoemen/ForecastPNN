from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

reporting_data = lambda x: x[0]
class ReportingDataset(Dataset):
    ## Theoretically, should contain covariates for date too, return tuple of matrix and covariates as well as label at each iteration

    def __init__(self, df, max_val, triangle = True, past_units=40, max_delay=40, future_obs = 0, device = "mps", vector_y = False, dow = False, return_number_obs = False):
        """
        Initialize the dataset with a start and end date.
        The dataset will generate matrices for each date within this range.
        
        Parameters:
        - start_date: The start date for generating matrices.
        - end_date: The end date for generating matrices.
        - past_days: Number of past days to consider for each matrix.
        - max_delay: Maximum delay to consider for each matrix.
        """
        if isinstance(df, pd.DataFrame):
            self.df = np.array(df, dtype = np.float32)
        else:
            self.df = df
        self.past_units = past_units
        self.max_delay = max_delay
        self.device = device
        self.triangle = triangle
        self.max_val = max_val
        self.future_obs = future_obs
        self.vector_y = vector_y
        self.dow = dow
        self.start_date = "2013-01-01"
        self.return_number_obs = return_number_obs

    def get_length(self):
        return self.df.shape[0]

    def __len__(self):
        # Calculate the number of days between 60 days after start_date and 46 days before end_date
        return len(self.df) - (self.past_units-1) - (self.max_delay-1)
    
    def __getitem__(self, idx):
        # Calculate the date for the current iteration, considering the adjusted range
        idx += self.past_units-1
        assert idx < len(self.df), "Index out of range"

        # Generate the matrix for the current date
        if self.dow:
            matrix, dow_val, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay, future_obs=self.future_obs, vector_y = self.vector_y, dow=self.dow)
            dow_val = torch.tensor(dow_val).to(self.device)
        else:
            matrix, label = reporting_data(self.df, idx=idx, past_units=self.past_units, max_delay=self.max_delay, future_obs=self.future_obs, vector_y = self.vector_y, dow=self.dow)
        
        # Convert the matrix to a PyTorch tensor
        tensor = torch.from_numpy(matrix)
        tensor = tensor.to(device=self.device)

        if not self.triangle: # sum
            tensor = torch.sum(tensor, dim = 1)
        
        # Compute the sum of the delays for the current date (row sum)
        label = torch.tensor([label]).to(self.device)
        if self.return_number_obs:
            num_obs = tensor.sum(axis = 1)[-(1+self.future_obs)].clone() # probably wrong
            label = (label, num_obs)
        if self.dow:
            return (tensor/self.max_val, dow_val), label 
        return tensor/self.max_val, label
        #return tensor, label
