import os

import pytest

from ml_project.config import Config
from ml_project.utils import get_project_root


@pytest.fixture(scope="package")
def config_base():

    return Config(historic_or_production_data='historic',
                  local_or_deployed='local',
                  aux_cols=[],
                  target_col='survived',
                  cont_cols=['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare'],
                  cat_cols=['sex', 'pclass'],
                  data_filepath=os.path.join(get_project_root(), "deployment/tests/test_data/titanic.parquet"),
                  export_filepath=os.path.join(get_project_root(), "deployment/tests/test_data/model_titanic.pkl"),
                  n_hyperparameter_optimisation_runs=1,
                  mlflow_experiment='exp1',
                  drift_dashboard_filepath="output/reports/",
                  )
