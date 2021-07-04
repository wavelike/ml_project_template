import pandas as pd
import pytest

from ml_project.config import Config
from ml_project.modelling_process.modelling_process import run_cross_validation_loop, run_hyperparameter_optimisation_cv_loop


@pytest.fixture
def data():

    # multiply each list with 3 in order to have each unique value in each potential fold in the cv runs
    return pd.DataFrame({
        'cont1': [-1, 2, 5, 7]*3,
        'cont2': [-3, 1, 6, 9]*3,
        'target': [0, 1, 1, 0]*3,
    })


@pytest.fixture
def config():

    return Config(
        historic_or_production_data='historic',
        local_or_deployed='local',
        target_col='target',
        cont_cols=['cont1', 'cont2'],
        cat_cols=[],
        aux_cols=[],
        data_filepath='', # not relevant for this text
        n_folds=3,
        optimisation_metric='acc_avg',
        n_hyperparameter_optimisation_runs=3,
    )


@pytest.fixture
def hyperparameters():

    return {
        'max_depth': 3,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
    }


def test_run_cross_validation_loop(config, data, hyperparameters):

    cv_metrics = run_cross_validation_loop(config, data, hyperparameters)

    acc_avg_in_cv_metrics = 'acc' in cv_metrics
    metric_values_correct_length = all([len(values) == config.n_folds for values in cv_metrics.values()])

    assert all([acc_avg_in_cv_metrics, metric_values_correct_length])


def test_run_hyperparameter_optimisation_cv_loop(config, data):

    leaderboard = run_hyperparameter_optimisation_cv_loop(config, data)

    leaderboard_has_correct_length = len(leaderboard) == config.n_hyperparameter_optimisation_runs
    leaderboard_contains_opt_metric = config.optimisation_metric in leaderboard.columns

    assert all([leaderboard_has_correct_length, leaderboard_contains_opt_metric])