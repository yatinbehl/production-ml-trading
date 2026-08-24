import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models import (
    create_logistic_model,
    create_random_forest_model,
    get_random_forest_feature_importance,
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


def test_random_forest_feature_importance():
    model = create_random_forest_model()

    X = [
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50],
        [6, 60],
        [7, 70],
        [8, 80],
        [9, 90],
        [10, 100],
        [11, 110],
        [12, 120],
        [13, 130],
        [14, 140],
        [15, 150],
        [16, 160],
        [17, 170],
        [18, 180],
        [19, 190],
        [20, 200],
        [21, 210],
        [22, 220],
        [23, 230],
        [24, 240],
        [25, 250],
        [26, 260],
        [27, 270],
        [28, 280],
        [29, 290],
        [30, 300],
    ]

    y = [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ]

    model.fit(X, y)

    importance = get_random_forest_feature_importance(
        model,
        ["feature_1", "feature_2"],
    )

    assert len(importance) == 2
    assert importance.index[0] in ["feature_1", "feature_2"]
    assert importance.sum() == pytest.approx(1.0)
