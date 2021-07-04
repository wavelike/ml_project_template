import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import CatTargetDriftTab, DataDriftTab, ProbClassificationPerformanceTab

logger = logging.getLogger('standard')


def create_drift_dashboard(reference_data: pd.DataFrame, reference_data_predictions: pd.Series, reference_data_predictions_probas: pd.DataFrame,
                           production_data: pd.DataFrame, production_data_predictions: pd.Series, production_data_predictions_probas: pd.DataFrame,
                           target_col: str, cont_cols: List[str], export_folderpath: Optional[str]):
    """
    Creates evidently plots for comparison of feature distribution and model predictions between reference (training) and production data
    """

    reference_data['prediction'] = reference_data_predictions.values
    production_data['prediction'] = production_data_predictions.values

    reference_data['target'] = reference_data[target_col]
    production_data['target'] = production_data[target_col]

    reference_data['result'] = reference_data['prediction']
    production_data['result'] = production_data['prediction']

    reference_data = pd.concat([reference_data, reference_data_predictions_probas], axis=1)
    production_data = pd.concat([production_data, production_data_predictions_probas], axis=1)

    # The 'CatTargetDriftTab' report needs same sample sizes of both datasets
    min_length = min(len(reference_data), len(production_data))
    reference_data = reference_data.sample(min_length)
    production_data = production_data.sample(min_length)

    column_mapping: Dict[str, Union[str, List]] = dict()
    column_mapping['target'] = target_col
    column_mapping['numerical_features'] = cont_cols

    # data drift and target drift analysis
    drift_report = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
    drift_report.calculate(reference_data=reference_data, current_data=production_data, column_mapping=column_mapping)
    if export_folderpath is not None:
        export_filepath = os.path.join(export_folderpath, "data_target_drift_report.html")
        drift_report.save(export_filepath)
    else:
        logger.warning("Warning: drift report was not exported since no 'export_filepath' is specified")

    # model performance drift analysis
    column_mapping['prediction'] = reference_data[target_col].unique().tolist()
    drift_report = Dashboard(tabs=[ProbClassificationPerformanceTab])
    drift_report.calculate(reference_data=reference_data, current_data=production_data, column_mapping=column_mapping)
    if export_folderpath is not None:
        export_filepath = os.path.join(export_folderpath, "prediction_drift_report.html")
        drift_report.save(export_filepath)
    else:
        logger.warning("Warning: drift report was not exported since no 'export_filepath' is specified")
