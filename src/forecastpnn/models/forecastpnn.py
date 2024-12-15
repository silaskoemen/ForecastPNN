import torch.nn as nn
import torch
from forecastpnn.distributions.NegativeBinomial import NegBin as NB


class ForecastPNN(nn.Module):
    def __init__(self, past_units = 45, n_layers = 3, hidden_units = [12, 8, 4]):
        super().__init__()
        self.past_units = past_units
        self.attfc1 = nn.Linear(self.past_units, self.past_units)
        self.attfc2 = nn.Linear(self.past_units, self.past_units)
        self.attfc3 = nn.Linear(self.past_units, self.past_units)
        self.attfc4 = nn.Linear(self.past_units, self.past_units)
        # Should iterate over n_layers for more robust solution and make ModuleList
        self.fc3 = nn.Linear(past_units, hidden_units[0])
        self.fc4 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc5 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fcnb = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.lnorm1, self.lnorm2, self.lnorm3, self.lnorm4 = nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1])
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.2), nn.Dropout(0.2), nn.Dropout(0.1)
        self.softplus = nn.Softplus()
        self.relu, self.silu = nn.ReLU(), nn.SiLU()
    
    def forward(self, x, prev):
        if x.size()[-1] != self.past_units:
            x = torch.squeeze(x)
        if len(prev.size()) > 1:
            prev = torch.squeeze(prev)
        x = self.silu(self.fc3(x))
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcnb(x)
        x[:, 0] *= self.const
        x[:, 1] *= self.const**2
        x[:, 0] += prev
        dist = NB(lbda = self.softplus(x[:, 0]), phi = self.softplus(x[:, 1])+1e-5)
        #dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
