from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models import (
    create_logistic_model,
    create_random_forest_model,
)


def test_create_logistic_model():
    model = create_logistic_model()

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_create_random_forest_model():
    model = create_random_forest_model()

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.max_depth == 5
    assert model.min_samples_leaf == 10
    assert model.random_state == 42
