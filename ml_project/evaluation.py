import logging
from typing import Any, List, Tuple

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from ml_project.config import Config

logger = logging.getLogger('standard')


def evaluate_predictions(config, true_labels: pd.Series, prediction_data: pd.Series, prediction_probas: pd.DataFrame):

    metrics = {}

    # Accuracy, Recall, Precision
    metrics['acc'] = accuracy_score(y_true=true_labels, y_pred=prediction_data)

    metrics['prec'] = precision_score(y_true=true_labels, y_pred=prediction_data)
    metrics['rec'] = recall_score(y_true=true_labels, y_pred=prediction_data)
    metrics['auc'] = roc_auc_score(y_true=true_labels, y_score=prediction_probas[1])

    logger.info(f"Prediction metrics: ")
    logger.info(metrics)

    return metrics


def evaluate_model(config: Config, model: Any, feature_names: List[str], evaluation_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info(f"Model evaluation")

    # Feature importances of final model
    feature_importances, shap_feature_importances = get_feature_importances(model, feature_names, evaluation_data)
    logger.info("Feature importances:")
    logger.info(feature_importances)

    if shap_feature_importances is not None:
        logger.info("Shap Feature importances:")
        logger.info(shap_feature_importances)

    return feature_importances, shap_feature_importances


def get_model_feature_importances(model, feature_names: List[str]) -> pd.DataFrame:

    sorted_feature_importances = sorted(list(zip(feature_names, model.feature_importances_)), key=lambda x: -x[1])
    feature_importances = pd.DataFrame(sorted_feature_importances, columns=['feature', 'importance'])

    return feature_importances


def get_shap_feature_importances(model, feature_names: List[str], evaluation_data: pd.DataFrame) -> pd.DataFrame:

    if type(model) in [RandomForestClassifier]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = None

    if explainer is not None:
        shap_values = pd.DataFrame(explainer.shap_values(evaluation_data)[1], columns=feature_names)
        feature_importances = shap_values.abs().mean().sort_values(ascending=False).reset_index()
        feature_importances.columns = ['feature', 'importance']
    else:
        logger.warning(f"shap feature importances currently not supported for model of type {type(model)}")
        feature_importances = None

    return feature_importances


def get_feature_importances(model, feature_names: List[str], evaluation_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    feature_importances = get_model_feature_importances(model, feature_names)
    shap_feature_importances = get_shap_feature_importances(model, feature_names, evaluation_data)

    return feature_importances, shap_feature_importances