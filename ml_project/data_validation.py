import pandas as pd
import pandera as pa
from pydantic import BaseModel

from ml_project.config import Config


class ProductionData(BaseModel):

    pclass: int
    sex: str
    age: float # TODO: int?
    siblings_spouses_aboard: int
    parents_children_aboard: int
    fare: float


# define schema for raw data
raw_data_schema = pa.DataFrameSchema({
    "survived": pa.Column(int, nullable=False, required=False, checks=pa.Check.isin([0, 1])),
    "pclass": pa.Column(int, nullable=True, required=True, checks=[pa.Check.ge(1), pa.Check.le(3)]),
    "sex": pa.Column(str, nullable=True, required=True, checks=pa.Check.isin(['male', 'female'])),
    "age": pa.Column(float, nullable=True, required=True, checks=[pa.Check.ge(0.), pa.Check.le(100.)]),
    "siblings_spouses_aboard": pa.Column(int, nullable=True, required=True, checks=[pa.Check.ge(0), pa.Check.le(5)]),
    "parents_children_aboard": pa.Column(int, nullable=True, required=True, checks=[pa.Check.ge(0), pa.Check.le(5)]),
    "fare": pa.Column(float, nullable=True, required=True, checks=[pa.Check.ge(7.), pa.Check.le(263.)]),
})

# define schema for engineered data
engineered_data_schema = pa.DataFrameSchema({
    "relatives_aboard": pa.Column(int, nullable=True, required=True, checks=[pa.Check.ge(0), pa.Check.le(10)]),
})


def validate_raw_data_per_instance(raw_data: pd.DataFrame):

    raw_data_schema(raw_data)


def validate_engineered_data_per_instance(engineered_data: pd.DataFrame):

    engineered_data_schema(engineered_data)



def validate_engineered_data_as_batch(config: Config, engineered_data: pd.DataFrame) -> pd.DataFrame:

    # TODO: Apply great_expectations

    pass