import pandas as pd
from epiweeks import Week
import torch
from forecastpnn.torch_utils.dataset import ReportingDataset
from loguru import logger
import numpy as np


def epi_weeks_date_range(min_week: Week, max_week: Week) -> pd.date_range:
    """Function to create a date range based on the minimum and maximum epi weeks.

    Args
    ----
    `min_week` [str]: Minimum epi week.
    `max_week` [str]: Maximum epi week.

    Returns
    -------
    weeks_list [list]: Date range based on the minimum and maximum epi weeks.
    """
    weeks_list = []
    current_week = min_week
    while current_week <= max_week:
        weeks_list.append(current_week)
        current_week = current_week + 1
    return weeks_list


def verify_weekly_data(
    dataset: pd.DataFrame, reference_col: str, time_format: str = "%Y-%m-%d"
) -> bool:
    """Function to verify if the input data is indexed by week.

    Args
    ----
    `dataset` [pd.DataFrame]: dataframe, organized by observations per day or week.
    `reference_col` [str]: Column to use as reference for the dataset.

    Returns
    -------
    [bool]: Whether the input data is indexed by week.
    """
    if pd.api.types.is_string_dtype(dataset[reference_col]):
        try:
            dataset[reference_col] = pd.to_datetime(
                dataset[reference_col], format=time_format
            )
            dataset[reference_col] = dataset[reference_col].apply(
                lambda x: Week.fromdate(x)
            )
        except ValueError:
            raise ValueError(
                "Reference column cannot be converted to epiweeks.Week format {}".format(
                    time_format
                )
            )
    elif pd.api.types.is_datetime64_any_dtype(dataset[reference_col]):
        try:
            dataset[reference_col] = dataset[reference_col].apply(
                lambda x: Week.fromdate(x)
            )
        except ValueError:
            raise ValueError(
                "Reference column cannot be converted to epiweeks.Week format {}".format(
                    time_format
                )
            )
    elif dataset[reference_col].dtype == Week:
        return dataset
    else:
        raise ValueError(
            "Reference column of weekly data is not string, pd.datetime or epiweeks.Week"
        )

    return dataset


def get_dataset_days(
    dataset: pd.DataFrame,
    reference_col: str,
    past_units: int = 12,
    return_df: bool = False,
    dow: bool = True,
    time_features: bool = False,
) -> torch.utils.data.Dataset:
    """Function to transform a given dataframe to a torch dataset.

    Args
    ----
    `dataset` [pd.DataFrame]: dataframe, organized by observations per day or week.
    `reference_col` [str]: Column to use as reference for the dataset.
    `past_units` [int]: Number of past units to consider for predictions.
    `return_df` [bool]: Whether to return the dataset as a dataframe.

    Returns
    -------
    [torch.utils.data.Dataset | pd.DataFrame]: Dataset to be used for training.
    """
    # Output data is indexed by day, group by each day and find total counts
    dataset = dataset.groupby(reference_col).size().to_frame(name="count")
    # Fill missing days with 0
    dataset = (
        dataset.reindex(
            pd.date_range(
                start=dataset.index.min(), end=dataset.index.max(), freq="D"
            )
        )
        .fillna(0)
        .astype(int)
    )
    if time_features:
        dataset['day_of_year'] = dataset.index.dayofyear
        # Transform the day of the year using sine and cosine
        dataset['sin_day_of_year'] = np.sin(2 * np.pi * dataset['day_of_year'] / 365)
        dataset['cos_day_of_year'] = np.cos(2 * np.pi * dataset['day_of_year'] / 365)
        dataset.drop(columns=['day_of_year'], inplace=True)

        dataset['day_of_week'] = dataset.index.dayofweek

        # Transform the day of the week using sine and cosine
        dataset['sin_day_of_week'] = np.sin(2 * np.pi * dataset['day_of_week'] / 7)
        dataset['cos_day_of_week'] = np.cos(2 * np.pi * dataset['day_of_week'] / 7)
        dataset.drop(columns=['day_of_week'], inplace=True)
    if return_df:
        return dataset
    else:
        return ReportingDataset(dataset, past_units=past_units, time_features=True)


def get_dataset_weeks(
    dataset: pd.DataFrame,
    reference_col: str,
    past_units: int = 12,
    return_df: bool = False,
    time_format: str = "%Y-%m-%d"
) -> torch.utils.data.Dataset:
    """Function to transform a given dataframe to a torch dataset.

    Args
    ----
    `dataset` [pd.DataFrame]: dataframe, organized by observations per day or week.
    `reference_col` [str]: Column to use as reference for the dataset.
    `past_units` [int]: Number of past units to consider for predictions.
    `return_df` [bool]: Whether to return the dataset as a dataframe.

    Returns
    -------
    [torch.utils.data.Dataset | pd.DataFrame]: Dataset to be used for training.
    """
    # Input data is indexed by week, group by each week and find total counts
    dataset = verify_weekly_data(dataset, reference_col, time_format)
    dataset = dataset.groupby(reference_col).size().to_frame(name="count")
    # Fill missing weeks with 0
    dataset = (
        dataset.reindex(
            epi_weeks_date_range(
                min_week=dataset.index.min(), max_week=dataset.index.max()
            )
        )
        .fillna(0)
        .astype(int)
    )
    dataset.index.name = None
    if return_df:
        return dataset
    else:
        return ReportingDataset(dataset, past_units=past_units, time_features=False)


def get_dataset_days_to_weeks(
    dataset: pd.DataFrame,
    reference_col: str,
    past_units: int = 12,
    return_df: bool = False
) -> torch.utils.data.Dataset:
    """Function to transform a given dataframe to a torch dataset.

    Args
    ----
    `dataset` [pd.DataFrame]: dataframe, organized by observations per day or week.
    `reference_col` [str]: Column to use as reference for the dataset.
    `past_units` [int]: Number of past units to consider for predictions.
    `return_df` [bool]: Whether to return the dataset as a dataframe.

    Returns
    -------
    [torch.utils.data.Dataset | pd.DataFrame]: Dataset to be used for training.
    """
    # Add weekly dimension to data
    dataset["week_ref"] = dataset[reference_col].apply(
        lambda x: Week.fromdate(x)
    )
    # Output data is indexed by week, group by each week and find total counts
    dataset = dataset.groupby("week_ref").size().to_frame(name="count")
    # Fill missing weeks with 0
    dataset = dataset.reindex(
        epi_weeks_date_range(
            min_week=dataset.index.min(), max_week=dataset.index.max()
        )
    ).fillna(0)
    dataset.index.name = None
    if return_df:
        return dataset
    else:
        return ReportingDataset(dataset, past_units=past_units, time_features=False)


def get_dataset(
    dataset: pd.DataFrame,
    reference_col: str,
    weeks_in: bool = False,
    weeks_out: bool = True,
    past_units: int = 12,
    return_df: bool = False,
    time_format: str = "%Y-%m-%d",
    dow: bool = True,
    filter_year_min: int = None,
    filter_year_max: int = None,
    time_features: bool = False,
) -> torch.utils.data.Dataset:
    """Function to transform a given dataframe to a torch dataset.

    Args
    ----
    `dataset` [pd.DataFrame]: dataframe, organized by observations per day or week.
    `reference_col` [str]: Column to use as reference for the dataset.
    `weeks_in` [bool]: Whether the input data has observations indexed by week.
    `weeks_out` [bool]: Whether the output data should have total number of cases per day or week.
    `past_units` [int]: Number of past units to consider for predictions.
    `return_df` [bool]: Whether to return the dataset as a dataframe.

    Returns
    -------
    [torch.utils.data.Dataset | pd.DataFrame]: Dataset to be used for training.
    """
    assert reference_col in dataset.columns, "Reference column not found in dataset."
    # Parse reference column to datetime format, only if daily
    if weeks_in:
        assert (
            weeks_out
        ), "If incoming data is indexed by weeks, output data cannot be returned by day."
    if weeks_out and dow:
        logger.warning(
            "If output data is indexed by weeks, day of the weeks cannot be used as feature, so keyword dow will be ignored."
        )
        dow = False

    if weeks_in:
        return get_dataset_weeks(
            dataset,
            reference_col,
            past_units=past_units,
            return_df=return_df
        )
    else:
        try:
            dataset[reference_col] = pd.to_datetime(
                dataset[reference_col], format=time_format, errors="coerce"
            )
        except ValueError:
            raise ValueError(
                "Reference column cannot be converted to datetime format {}".format(
                    time_format
                )
            )
        dataset = dataset.dropna(subset=[reference_col])
        if filter_year_min:
            dataset = dataset[dataset[reference_col].dt.year >= filter_year_min]
        if filter_year_max:
            dataset = dataset[dataset[reference_col].dt.year <= filter_year_max]

        if weeks_out:
            return get_dataset_days_to_weeks(
                dataset, reference_col, past_units=past_units, return_df=return_df
            )
        else:
            return get_dataset_days(
                dataset, reference_col, past_units=past_units, return_df=return_df, time_features=time_features
            )


""" Could use to find units of maximum value, return with dataset and then parse to NN as self.const
counter = len(str(max_number))
    
# Calculate the nearest unit of length based on the counter
nearest_unit = 10 ** (counter - 1)
"""
