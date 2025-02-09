# %%
import os
from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent / 'src'))
from forecastpnn.utils.data_functions import get_dataset_percentage_change
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler as SRS
from sklearn.model_selection import train_test_split as TTS
import torch
import random
import numpy as np
torch.use_deterministic_algorithms(True) # reproducibility
import pandas as pd

from forecastpnn.utils.train_utils import SubsetSampler as SS
from forecastpnn.utils.constants import RANDOM_SEED

WEEKS = False
PAST_UNITS = 42
BATCH_SIZE = 64
RANDOM_SPLIT = False
DEVICE = "mps"

data = pd.read_csv('../data/derived/DENGSP.csv', index_col=0)
#weekly_df = get_dataset(data, 'DT_SIN_PRI', weeks_in=False, weeks_out=WEEKS, past_units=PAST_UNITS, return_df=True, filter_year_min=2013, filter_year_max=2020)
dl = get_dataset_percentage_change(data, 'DT_SIN_PRI', weeks_in=False, weeks_out=WEEKS, past_units=PAST_UNITS, return_df=False, filter_year_min=2013, filter_year_max=2020, time_features=False)

#n_obs_40pu = len(dataset) # 2922 total dates, -39-39 for past_units and max_delay ->2844
## Define train and test indices
if RANDOM_SPLIT:
    all_idcs = range(dl.__len__())
    train_idcs, test_idcs = TTS(all_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
    train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
    #train_idcs, test_idcs = [*range(600), *range(950, dataset.__len__())], [*range(600, 950)]
    VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
else:
    if WEEKS: # could also do random split, for now last indices as test
        train_idcs, test_idcs = range(300), range(300, dl.__len__())
        train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
    else: 
        train_idcs, test_idcs = range(int(0.75*dl.__len__())), range(int(0.75*dl.__len__()), dl.__len__()) # 2844 total obs - 711 test, still 25% even without random split, last outbreak 2353
        train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
        
## Define generator so sampling during training is deterministic and reproducible
g = torch.Generator()
g.manual_seed(RANDOM_SEED)
train_sampler, val_sampler, test_sampler = SRS(train_idcs, generator=g), SRS(val_idcs), SS(test_idcs)
train_loader, val_loader, test_loader = DataLoader(dl, batch_size=BATCH_SIZE, sampler=train_sampler), DataLoader(dl, batch_size=VAL_BATCH_SIZE, sampler=val_sampler, shuffle=False), DataLoader(dl, batch_size=TEST_BATCH_SIZE, sampler=test_sampler, shuffle=False)

## Function to reset the sampler so each training run uses same order of observations for reproducibility
## Possible to define s.t. returns train_loader, but bc in notebook, possible to define globally
def regen_data():
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    global train_loader
    train_loader = DataLoader(dl, batch_size=BATCH_SIZE, sampler=SRS(train_idcs, generator=g))

def set_seeds(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

set_seeds(RANDOM_SEED)
#%%
next(iter(train_loader))
# %%
import forecastpnn.models.forecastpnn
import importlib
importlib.reload(forecastpnn.models.forecastpnn)
importlib.reload(forecastpnn.models)
importlib.reload(forecastpnn.utils.train_utils)
from forecastpnn.utils.train_utils import train, EarlyStopper
from forecastpnn.models.forecastpnn import TCNInception, TCNForecaster
set_seeds(RANDOM_SEED)

regen_data() # reset samplers so each training run is reproducible
early_stopper = EarlyStopper(patience=30, past_units=PAST_UNITS, weeks=WEEKS, future_obs=0)
forecaster = TCNForecaster(past_units=PAST_UNITS)
#forecast_pnn = PNNSumDaily(past_units=PAST_UNITS, max_delay=MAX_DELAY)  # "mse_real"
train(forecaster, num_epochs=500, train_loader=train_loader, val_loader=val_loader, early_stopper=early_stopper, loss_fct="nll", device = DEVICE)
## Load best set of weights on test/validation set
forecaster.load_state_dict(torch.load(f".weights/weights-{PAST_UNITS}-{'week' if WEEKS else 'day'}{'-rec' if not RANDOM_SPLIT else ''}{'-dow' if False else ''}"))
# %%
importlib.reload(forecastpnn.utils.plotting)
from forecastpnn.utils.plotting import plot_entire_confints_percentage
set_seeds(RANDOM_SEED)
plot_entire_confints_percentage(dl, forecaster, weeks = WEEKS, random_split = RANDOM_SPLIT, test_idcs=test_idcs, xlims=[2500, 2600])

# %%
STEPS_AHEAD = 21
importlib.reload(forecastpnn.utils.data_functions)
from forecastpnn.utils.data_functions import get_dataset_percentage_change
dl = get_dataset_percentage_change(data, 'DT_SIN_PRI', weeks_in=False, weeks_out=WEEKS, past_units=PAST_UNITS, return_df=False, filter_year_min=2013, filter_year_max=2020, time_features=False, steps_ahead=STEPS_AHEAD)

#n_obs_40pu = len(dataset) # 2922 total dates, -39-39 for past_units and max_delay ->2844
## Define train and test indices
if RANDOM_SPLIT:
    all_idcs = range(dl.__len__())
    train_idcs, test_idcs = TTS(all_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
    train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
    #train_idcs, test_idcs = [*range(600), *range(950, dataset.__len__())], [*range(600, 950)]
    VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
else:
    if WEEKS: # could also do random split, for now last indices as test
        train_idcs, test_idcs = range(300), range(300, dl.__len__())
        train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
    else: 
        train_idcs, test_idcs = range(int(0.75*dl.__len__())), range(int(0.75*dl.__len__()), dl.__len__()) # 2844 total obs - 711 test, still 25% even without random split, last outbreak 2353
        train_idcs, val_idcs = TTS(train_idcs, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
        VAL_BATCH_SIZE, TEST_BATCH_SIZE = len(val_idcs), len(test_idcs)
        
## Define generator so sampling during training is deterministic and reproducible
g = torch.Generator()
g.manual_seed(RANDOM_SEED)
train_sampler, val_sampler, test_sampler = SRS(train_idcs, generator=g), SRS(val_idcs), SS(test_idcs)
train_loader, val_loader, test_loader = DataLoader(dl, batch_size=BATCH_SIZE, sampler=train_sampler), DataLoader(dl, batch_size=VAL_BATCH_SIZE, sampler=val_sampler, shuffle=False), DataLoader(dl, batch_size=TEST_BATCH_SIZE, sampler=test_sampler, shuffle=False)

# %%
import forecastpnn.models.forecastpnn
import importlib
importlib.reload(forecastpnn.models.forecastpnn)
importlib.reload(forecastpnn.models)
importlib.reload(forecastpnn.utils.train_utils)
from forecastpnn.utils.train_utils import train_multistep, EarlyStopper
from forecastpnn.models.forecastpnn import TCNForecaster
set_seeds(RANDOM_SEED)

regen_data() # reset samplers so each training run is reproducible
early_stopper = EarlyStopper(patience=25, past_units=PAST_UNITS, weeks=WEEKS, future_obs=0)
forecaster = TCNForecaster(past_units=PAST_UNITS)
#forecast_pnn = PNNSumDaily(past_units=PAST_UNITS, max_delay=MAX_DELAY)  # "mse_real"
train_multistep(forecaster, num_epochs=150, train_loader=train_loader, val_loader=val_loader, early_stopper=early_stopper, loss_fct="nll", device = DEVICE)
## Load best set of weights on test/validation set
forecaster.load_state_dict(torch.load(f".weights/weights-{PAST_UNITS}-{'week' if WEEKS else 'day'}{'-rec' if not RANDOM_SPLIT else ''}{'-dow' if False else ''}"))
# %%
importlib.reload(forecastpnn.utils.plotting)
from forecastpnn.utils.plotting import plot_confints_forecast_with_updated_inputs
set_seeds(RANDOM_SEED)
plot_confints_forecast_with_updated_inputs(dl, forecaster, weeks = WEEKS, idx=770, steps_ahead=STEPS_AHEAD)

# %%
