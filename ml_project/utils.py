import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import yaml


def set_mlflow_experiment(experiment_name: str, sqlite_uri: Optional[str] = None):
    """
    Sets tracking_uri for the mlflow metadata database and sets experiment_name
    """

    if sqlite_uri is None:
        sqlite_uri = f'sqlite:///{os.path.join(get_project_root(), "data/mlflow.db")}'


    mlflow.set_tracking_uri(sqlite_uri)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=os.path.join(get_project_root(), f'data/mlruns/{experiment_name}'))
    mlflow.set_experiment(experiment_name)


def setup_logging(name: str) -> logging.Logger:

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    return logger


def get_model_artifacts_filepath() -> str:
    """
    Helper function to retrieve the model artifacts filepath from the 'project_config.yaml' file
    """

    project_configs_filepath = os.path.join(get_project_root(), "project_config.yaml")
    with open(project_configs_filepath, 'r') as stream:
        project_configs = yaml.safe_load(stream)

    model_objects_filepath = os.path.join(get_project_root(), project_configs['MODEL_OBJECTS_FILEPATH'])

    return model_objects_filepath


def get_project_root() -> str:
    """
    Note: When moving this method to another module or changing the nesting layer of the current module relative to the project root, remove or add the respective amount of '.parent' chain commands
    """

    return str(Path(__file__).parent.parent)