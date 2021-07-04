import logging
from typing import List, Tuple

import pandas as pd

from ml_project.config import Config
from ml_project.feature_engineering.feature_processes import Feature1Feature2Sum

logger = logging.getLogger('standard')


def get_feature_processes(config: Config):

    feature_processes = [
        Feature1Feature2Sum(feature1_col='siblings_spouses_aboard',
                            feature2_col='parents_children_aboard',
                            column_name='relatives_aboard',
                            ),
    ]

    return feature_processes


def execute_feature_engineering(config: Config, data: pd.DataFrame, feature_processes: List) -> Tuple[pd.DataFrame, List]:
    """
    Executes each process in 'feature_processes' and adds the resulting feature to 'data'
    """

    for feature_process in feature_processes:
        logger.info(f"Engineering feature '{feature_process.column_name}'")
        data = feature_process.execute(data)

    engineered_feature_columns = [feature_process.column_name for feature_process in feature_processes]

    return data, engineered_feature_columns