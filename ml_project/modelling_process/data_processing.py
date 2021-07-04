from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ml_project.config import Config


@dataclass
class PreprocessingObjects:

    one_hot_encoders: Optional[Dict[str, OneHotEncoder]] = None
    features: Optional[List[str]] = None


def scale_data(train_data, validation_data):
    """
    Scales the columns of the dataset based on train_data
    :return:
    """

    #####
    # Scale data per column
    # TODO: scikit-learn standard scaler
    # TODO: scikit-learn minmax scaler
    #####

    return train_data, validation_data


def remove_outliers(train_data, validation_data):
    """
    Removes outliers based on train_data
    :return:
    """

    #####
    # Elliptic Envelope
    # TODO:
    #####

    #####
    # Isolation Forest
    # TODO:
    #####

    return train_data, validation_data


def apply_categorical_encoding(config: Config,
                               preprocessing_objects: PreprocessingObjects,
                               train_data: pd.DataFrame,
                               validation_data: Optional[pd.DataFrame] = None,
                               ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], PreprocessingObjects]:
    """
    Appl categorical encoding based on the train_data
    :param train_data:
    :param validation_data:
    :return:
    """

    #####
    # One-Hot-Encoding
    if preprocessing_objects.one_hot_encoders is None:
        preprocessing_objects.one_hot_encoders = {}

    for cat_feature in config.cat_cols:

        if cat_feature not in preprocessing_objects.one_hot_encoders:
            one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
            one_hot_encoder.fit(train_data[cat_feature].values.reshape(-1, 1))  # Reshape for single feature
            preprocessing_objects.one_hot_encoders[cat_feature] = one_hot_encoder
        else:
            one_hot_encoder = preprocessing_objects.one_hot_encoders[cat_feature]

        cat_values = one_hot_encoder.categories_[0].tolist()
        one_hot_encoded_column_names = [f"{cat_feature}_{category}" for category in cat_values]

        train_one_hot_encoded_columns = pd.DataFrame(one_hot_encoder.transform(train_data[cat_feature].values.reshape(-1, 1)),
                                                       columns=one_hot_encoded_column_names,
                                                       index=train_data.index)
        train_data = train_data.join(train_one_hot_encoded_columns, how='left')
        train_data = train_data.drop(columns=[cat_feature])

        if validation_data is not None:
            validation_one_hot_encoded_columns = pd.DataFrame(one_hot_encoder.transform(validation_data[cat_feature].values.reshape(-1, 1)),
                                                   columns=one_hot_encoded_column_names,
                                                   index=validation_data.index)
            validation_data = validation_data.join(validation_one_hot_encoded_columns, how='left')
            validation_data = validation_data.drop(columns=[cat_feature])

        #config.cat_cols = [cat_col for cat_col in config.cat_cols if cat_col != cat_feature]
        #config.cont_cols = config.cont_cols + one_hot_encoded_column_names

    #####

    #####
    # Label Encoding
    # TODO
    #####

    return train_data, validation_data, preprocessing_objects


def apply_dimensionality_reduction(train_data, validation_data):
    """
    Applys dimensionality reduction methods like PCA or LDA based on train_data
    :return:
    """

    #####
    # PCA
    # TODO: PCA
    #####

    #####
    # LDA
    # TODO: PCA
    #####

    return train_data, validation_data


def get_processed_data(config: Config, preprocessing_objects: Optional[PreprocessingObjects], train_x: pd.DataFrame, validation_x: Optional[pd.DataFrame] = None):

    if preprocessing_objects is None:
        preprocessing_objects = PreprocessingObjects()

    train_x, validation_x, preprocessing_objects = apply_categorical_encoding(config, preprocessing_objects, train_x, validation_x)
    train_x, validation_x = scale_data(train_x, validation_x)
    train_x, validation_x = remove_outliers(train_x, validation_x)
    train_x, validation_x = apply_dimensionality_reduction(train_x, validation_x)

    preprocessing_objects.features = train_x.columns.tolist()

    return train_x, validation_x, preprocessing_objects