""" File to test whether, given an input file, the number of counts
per day are"""

import pytest
import pandas as pd
from datetime import datetime
from epiweeks import Week

from forecastpnn.utils.data_functions import get_dataset


@pytest.fixture
def test_input_invalid_date_format():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', "2020-06-31"],
        'region': ["Sao Paulo", "Rio", "Minas Gerais", "Bahia", "Sao Paulo"],
    })


@pytest.fixture
def test_input_days():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-04'],
        'region': ["Sao Paulo", "Rio", "Minas Gerais", "Bahia", "Sao Paulo", "Rio"],
        'id': [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def test_input_weeks_string():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-03-01', '2020-03-02', '2020-03-02', '2020-01-02', '2020-01-04'],
        'region': ["Sao Paulo", "Rio", "Minas Gerais", "Bahia", "Sao Paulo", "Rio"],
        'id': [1, 2, 3, 4, 5, 6],
    })

@pytest.fixture
def test_input_weeks_date():
    return pd.DataFrame({
        'date': [datetime(2020, 1, 1), datetime(2020, 3, 1), datetime(2020, 3, 2), datetime(2020, 3, 2), datetime(2020, 1, 2), datetime(2020, 1, 4)],
        'region': ["Sao Paulo", "Rio", "Minas Gerais", "Bahia", "Sao Paulo", "Rio"],
        'id': [1, 2, 3, 4, 5, 6],
    })

@pytest.fixture
def test_input_weeks_week():
    return pd.DataFrame({
        'date': [Week(2020, 1), Week(2020, 10), Week(2020, 10), Week(2020, 10), Week(2020, 1), Week(2020, 1)],
        'region': ["Sao Paulo", "Rio", "Minas Gerais", "Bahia", "Sao Paulo", "Rio"],
        'id': [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def expected_output_days():
    df = pd.DataFrame({
        'date': [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4)],
        'count': [3, 2, 0, 1],
    }).set_index('date')
    df.index.name = None
    df.index.freq = 'D'
    return df


@pytest.fixture
def expected_output_day_to_weeks():
    df = pd.DataFrame({
        'date': [Week(2020, 1)],
        'count': [6],
    }).set_index('date')
    df.index.name = None
    df.index.freq = 'W'
    return df

@pytest.fixture
def expected_output_weeks():
    df = pd.DataFrame({
        'week': [
            Week(2020, 1),
            Week(2020, 2),
            Week(2020, 3),
            Week(2020, 4),
            Week(2020, 5),
            Week(2020, 6),
            Week(2020, 7),
            Week(2020, 8),
            Week(2020, 9),
            Week(2020, 10)],
        'count': [3,0,0,0,0,0,0,0,0,3],
    }).set_index('week')
    df.index.name = None
    df.index.freq = 'W'
    return df


class TestInputs():
    def test_invalid_input_date(self, test_input_invalid_date_format):
        with pytest.raises(ValueError) as excinfo:
            ds = get_dataset(test_input_invalid_date_format, reference_col='date')
        assert "Reference column cannot be converted to datetime format" in str(excinfo.value)

    def test_invalid_input_col(self, test_input_invalid_date_format):
        with pytest.raises(AssertionError) as excinfo:
            ds = get_dataset(test_input_invalid_date_format, reference_col='col')
        assert "Reference column not found in dataset." in str(excinfo.value)
    
    def test_valid_input_days(self, test_input_days, expected_output_days):
        ds = get_dataset(test_input_days, reference_col='date', weeks_in=False, weeks_out=False, return_df=True)
        assert isinstance(ds, pd.DataFrame)
        pd.testing.assert_frame_equal(ds, expected_output_days)
    
    def test_valid_input_day_to_week(self, test_input_days, expected_output_day_to_weeks):
        ds = get_dataset(test_input_days, reference_col='date', weeks_in=False, weeks_out=True, return_df=True)
        pd.testing.assert_frame_equal(ds, expected_output_day_to_weeks)
    
    def test_valid_input_weeks_string(self, test_input_weeks_string, expected_output_weeks):
        ds = get_dataset(test_input_weeks_string, reference_col='date', weeks_in=True, weeks_out=True, return_df=True)
        pd.testing.assert_frame_equal(ds, expected_output_weeks)

    def test_valid_input_weeks_date(self, test_input_weeks_date, expected_output_weeks):
        ds = get_dataset(test_input_weeks_date, reference_col='date', weeks_in=True, weeks_out=True, return_df=True)
        pd.testing.assert_frame_equal(ds, expected_output_weeks)
    
    def test_valid_input_weeks_week(self, test_input_weeks_week, expected_output_weeks):
        ds = get_dataset(test_input_weeks_week, reference_col='date', weeks_in=True, weeks_out=True, return_df=True)
        pd.testing.assert_frame_equal(ds, expected_output_weeks)