
import pandas as pd

from ml_project.config import Config
from ml_project.data_validation import raw_data_schema


def retrieve_from_parquet(filepath_parquet: str) -> pd.DataFrame:

    data = pd.read_parquet(filepath_parquet).iloc[:100]

    return data


def retrieve_historic_data(config: Config) -> pd.DataFrame:
    """
    Retrieves historical data from the relevant data source (files, databases, S3 buckets, etc.)
    """

    # from parquet file
    historic_data = retrieve_from_parquet(filepath_parquet=config.data_filepath)
    historic_data = historic_data[list(raw_data_schema.columns)] # only keep columns defined in the data_schema

    return historic_data


def process_historic_data_into_raw_data(config: Config, historic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processing of the historic data into raw data that adheres to a given schema (column names, column types, NaN handling, etc.).
    Raw data originating from historic data has the exact same schema as raw_data retrieved from production sources in order to allow unified further processing.
    """

    #####
    ### Column selection
    historic_data = historic_data[config.features + [config.target_col]]
    ###

    #####
    ### NaN imputing for continuous and categorical columns
    historic_data.loc[:, config.cat_cols] = historic_data[config.cat_cols].fillna('<missing>')
    historic_data.loc[:, config.cont_cols] = historic_data[config.cont_cols].fillna(historic_data[config.cont_cols].min() - 999999)
    #####

    return historic_data


