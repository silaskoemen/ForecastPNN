# General TODOs

- For time of day and of year, use transforms instead of single value with embedding (https://www.tensorflow.org/tutorials/structured_data/time_series)
- Define index as day of last observation, label is +1, or label is idx itself. Also, implement getitem with number of timesteps returned at once to calculate loss
- Compare single prediction of all future values vs iterative preds
- Use counts of neighboring regions as input (call transfer learning), might be able to spot how cross-influence
- ResNet, just predict how changes from previous value, give max_val to model and then predict change over last input, whether up or down only
- Levenberg Marquardt optimization (small dataset)
- Think about how to wrap data loading and definitions, maybe keep samplers as return, otherwise could keep in wrapper s.t. just have to call train and uses under the hood
- Wrapper first has model definition, then when calling train has train/test split arg and give dataset, if patience for early stop given (not None or -1) automatically create validation set, have prediction wrapper that only has to call model predict, auto re-feed for multiple steps
- LSTM with residual connection
- Include time of year (and for daily time of week) with sin/cos transform as features, enables nearby region as feature too bc input is already 2D
- Try out convolutional layer over only counts, residual connection, then fully connected/LSTM
- Add keywords to add time features, use LSTM, use convolutional or just Linear as hyperparameters
- Compare different model types and modeling choices, show variety and outperform others
- Add kw for number of units in the future, count up all losses and then backpropagate so model hopefully doesn't zero-predict