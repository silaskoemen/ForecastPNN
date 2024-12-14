# General TODOs

- For time of day and of year, use transforms instead of single value with embedding (https://www.tensorflow.org/tutorials/structured_data/time_series)
- Define index as day of last observation, label is +1, or label is idx itself. Also, implement getitem with number of timesteps returned at once to calculate loss
- Compare single prediction of all future values vs iterative preds
- Use counts of neighboring regions as input (call transfer learning), might be able to spot how cross-influence
- ResNet, just predict how changes from previous value, give max_val to model and then predict change over last input, whether up or down only
- Levenberg Marquardt optimization (small dataset)