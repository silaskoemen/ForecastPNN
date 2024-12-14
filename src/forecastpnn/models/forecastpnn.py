import torch.nn as nn
import torch
from forecastpnn.distributions.NegativeBinomial import NegBin as NB


## For summed (one-dimensional) input data
class ForecastPNN(nn.Module):
    def __init__(self, past_units = 45, n_layers = 3, hidden_units = [64, 32, 16]):
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
        self.fcpoi = nn.Linear(hidden_units[2], 1)
        self.fcnb = nn.Linear(hidden_units[2], 2)
        self.const = 10000 # because output is very large values, find scale and save as constant

        self.bnorm1, self.bnorm2, self.bnorm3, self.bnorm4 = nn.BatchNorm1d(num_features=past_units), nn.BatchNorm1d(num_features=hidden_units[0]), nn.BatchNorm1d(num_features=hidden_units[1]), nn.BatchNorm1d(num_features=hidden_units[2])
        self.lnorm1, self.lnorm2, self.lnorm3, self.lnorm4 = nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1]), nn.LayerNorm([1])
        self.attn1 = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(0.2), nn.Dropout(0.4), nn.Dropout(0.2)
        self.softplus = nn.Softplus()
        self.relu, self.silu = nn.ReLU(), nn.SiLU()
    
    def forward(self, x):
        #print(x.size())
        #x = x + self.pos_embed(x)
        x = torch.unsqueeze(x, -1)#.permute(0, 2, 1)
        #print(f"Before att layers: {x.size()}")
        x_add = x.clone()
        x = self.lnorm1(x)
        x = self.attn1(x, x, x, need_weights = False)[0]
        x = self.attfc1(x.permute(0, 2, 1))
        x = self.silu(x).permute(0, 2, 1)
        x = x + x_add
        x = x.permute(0, 2, 1) # [batch, past_units, 1] -> [batch, 1, past_units], so can take past_units
        x = torch.squeeze(x)
        x = self.silu(self.fc3(self.bnorm1(x)))
        x = self.drop1(x)
        x = self.silu(self.fc4(self.bnorm2(x)))
        x = self.drop2(x)
        x = self.silu(self.fc5(self.bnorm3(x)))
        x = self.drop3(x)
        x = self.fcnb(self.bnorm4(x))
        #dist = torch.distributions.Poisson(rate=1000*self.softplus(x))
        dist = NB(lbda = self.const*self.softplus(x[:, 0]), phi = self.const**2*self.softplus(x[:, 1])+1e-5)
        return torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
