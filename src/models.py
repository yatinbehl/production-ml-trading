import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )


def create_random_forest_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
def get_random_forest_feature_importance(model, feature_names):
    return pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
