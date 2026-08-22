import pandas as pd

from src.target import add_target


def test_add_target():
    data = pd.DataFrame(
        {
            "Close": [100, 102, 101, 104, 106, 110, 108],
        }
    )

    result = add_target(data, horizon=5)

    assert result.loc[0, "future_close"] == 110
    assert result.loc[0, "target"] == 1

    assert result.loc[1, "future_close"] == 108
    assert result.loc[1, "target"] == 1

    assert pd.isna(result.loc[2, "future_close"])
    assert pd.isna(result.loc[2, "target"])
