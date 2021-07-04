from typing import Optional

import pandas as pd


class Feature1Feature2Sum:
    """
    Calculates the sum of two input features
    """

    def __init__(self,
                 feature1_col: str,
                 feature2_col: str,
                 column_name: Optional[str] = None
                 ):

        self.feature1_col = feature1_col
        self.feature2_col = feature2_col

        if column_name is None:
            self.column_name = f"{feature1_col}_{feature2_col}_sum"
        else:
            self.column_name = column_name

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:

        data[self.column_name] = data[self.feature1_col] + data[self.feature2_col]

        return data


class Feature1Feature2Ratio:
    """
    Calculates the ratio of two input features
    """

    def __init__(self,
                 feature1_col: str,
                 feature2_col: str,
                 column_name: Optional[str] = None
                 ):

        self.feature1_col = feature1_col
        self.feature2_col = feature2_col

        if column_name is None:
            self.column_name = f"{feature1_col}_{feature2_col}_ratio"
        else:
            self.column_name = column_name

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:

        data[self.column_name] = data[self.feature1_col] / data[self.feature2_col]

        return data


