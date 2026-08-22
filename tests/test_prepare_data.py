from src.prepare_data import prepare_ml_data, time_series_split
import pandas as pd

from src.prepare_data import prepare_ml_data


def test_prepare_ml_data():
    data = pd.DataFrame(
        {
            "daily_return": [0.01, None, -0.02],
            "momentum_5d": [0.05, 0.03, -0.01],
            "volatility_5d": [0.02, 0.01, 0.03],
            "target": [1, 0, 0],
        }
    )

    X, y = prepare_ml_data(data)

    assert len(X) == 2
    assert len(y) == 2
    assert X.columns.tolist() == [
        "daily_return",
        "momentum_5d",
        "volatility_5d",
    ]
    assert y.tolist() == [1, 0]
def test_time_series_split():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
        }
    )

    y = pd.Series([0, 1, 0, 1, 1])

    X_train, X_test, y_train, y_test = time_series_split(
        X,
        y,
        train_ratio=0.8,
    )

    assert X_train["feature"].tolist() == [1, 2, 3, 4]
    assert X_test["feature"].tolist() == [5]

    assert y_train.tolist() == [0, 1, 0, 1]
    assert y_test.tolist() == [1]
