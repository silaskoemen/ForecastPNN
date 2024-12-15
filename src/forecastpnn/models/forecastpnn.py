import torch.nn as nn
import torch
from forecastpnn.distributions.NegativeBinomial import NegBin as NB
from torch.distributions import Normal, StudentT


class ForecastPNN(nn.Module):
    def __init__(self, past_units = 45, n_layers = 3, hidden_units = [12, 8, 4]):
        super().__init__()
        self.past_units = past_units
        # Should iterate over n_layers for more robust solution and make ModuleList
        self.fc3 = nn.Linear(past_units, hidden_units[0])
        self.fc4 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[2], 2)
        self.fcn = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant
        self.fc_loc = nn.Linear(hidden_units[2], 1)
        self.fc_scale = nn.Linear(hidden_units[2], 1)

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.lnorm1, self.lnorm2, self.lnorm3, self.lnorm4 = nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1])
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.1), nn.Dropout(0.1), nn.Dropout(0.1)
        self.softplus = nn.Softplus()
        self.relu, self.silu = nn.ReLU(), nn.SiLU()
    
    def forward(self, x):
        if len(x.size()) > 1:
            x = torch.squeeze(x)
        x = self.silu(self.fc3(x))  # Take last timestep output
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcn(x)
        #x = self.fcnb(self.bnorm4(x))
        #x[:, 0] = self.const * x[:, 0] + p
        """ dist = NB(lbda = self.const * self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        #dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1) """
        dist = StudentT(df=self.past_units-1, loc = self.const * x[:, 0], scale = self.const/10*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        dist = Normal(loc = self.const * x[:, 0], scale = self.const/10*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        x = self.const*self.fc_final(self.bnorm4(x))
        return torch.squeeze(x)


class ForecastPNNDayMultiFeature(nn.Module):
    def __init__(self, past_units=30, n_features=5, hidden_size = 128, n_layers=3, hidden_units=[24, 12, 4]):
        super().__init__()
        self.past_units = past_units
        self.n_features = n_features
        
        # Modified LSTM layers to accept n_features as input_size
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcn = nn.Linear(hidden_units[2], 2)
        self.const = 10000
        
        # Rest of the layers
        self.bnorm1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bnorm2 = nn.BatchNorm1d(num_features=hidden_units[0])
        self.bnorm3 = nn.BatchNorm1d(num_features=hidden_units[1])
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

    def forward(self, x):
        if x.size()[-1] != self.n_features:
            x = torch.squeeze(x)
        x, _ = self.lstm1(x)
        x = self.silu(self.fc1(self.bnorm1(torch.squeeze(x[:, -1, :], 1))))
        x = self.drop1(x)
        x = self.silu(self.fc2(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc3(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcn(x)
        dist = NB(lbda = self.const * self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        dist = StudentT(df=self.past_units-1, loc=self.const * x[:, 0], 
                        scale=self.const/10*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)


class ForecastPNNDay(nn.Module):
    def __init__(self, past_units = 30, hidden_units = [24, 12, 4]):
        super().__init__()
        self.past_units = past_units
        # Should iterate over n_layers for more robust solution and make ModuleList
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=past_units, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=past_units, hidden_size=past_units, num_layers=1, batch_first=True)
        self.fc3 = nn.Linear(past_units, hidden_units[0])
        self.fc4 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcn = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.1), nn.Dropout(0.1), nn.Dropout(0.1)
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        
    def forward(self, x):
        """ if x.size()[-1] != self.past_units:
            x = torch.squeeze(x) """
        x_add = x.clone()
        x, (h_n, c_n) = self.lstm1(x)
        x, _ = self.lstm2(x, (h_n, c_n))
        x = x + x_add
        x = self.silu(self.fc3(torch.squeeze(x[:, -1, :])))
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcn(x)
        #x = self.fcnb(self.bnorm4(x))
        #x[:, 0] = self.const * x[:, 0] + p
        """ dist = NB(lbda = self.const * self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        #dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1) """
        dist = StudentT(df=self.past_units-1, loc = self.const * x[:, 0], scale = self.const/10*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)

