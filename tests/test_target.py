import pandas as pd
import pytest

from src.target import add_target


def test_add_target():
    df = pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                103,
                104,
                102,
                103,
                104,
                105,
                106,
            ]
        }
    )

    result = add_target(
        df,
        horizon=5,
        threshold=0.01,
    )

    # Row 0: 100 -> 102 = +2%, so target = 1
    assert result.loc[0, "future_close"] == 102
    assert result.loc[0, "future_return"] == pytest.approx(0.02)
    assert result.loc[0, "target"] == 1

    # Row 1: 101 -> 103 = about +1.98%, so target = 1
    assert result.loc[1, "target"] == 1

    # Last 5 rows do not yet have known future prices
    assert result["future_close"].tail(5).isna().all()
    assert result["future_return"].tail(5).isna().all()
    assert result["target"].tail(5).isna().all()
