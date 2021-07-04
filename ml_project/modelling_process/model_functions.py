import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_model(config, hyperparameters) -> RandomForestClassifier:

    # random_forest
    model = RandomForestClassifier(n_estimators=10,
                                   **hyperparameters,
                                   n_jobs=-1,
                                   )

    return model


def train_model(config, model: RandomForestClassifier, train_x: pd.DataFrame, train_y: pd.Series) -> RandomForestClassifier:

    model.fit(X=train_x,
              y=train_y,
              )

    return model


