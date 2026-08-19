import pandas as pd
import pytest

from src.features import (
    add_daily_return,
    add_momentum,
    add_volatility,
)


def test_add_daily_return():
    data = pd.DataFrame(
        {
            "Close": [100, 105, 102],
        }
    )

    result = add_daily_return(data)

    assert pd.isna(result.loc[0, "daily_return"])
    assert result.loc[1, "daily_return"] == pytest.approx(0.05)
    assert result.loc[2, "daily_return"] == pytest.approx(
        -0.0285714, abs=1e-6
    )


def test_add_momentum():
    data = pd.DataFrame(
        {
            "Close": [100, 105, 110, 115, 120, 125],
        }
    )

    result = add_momentum(data, window=5)

    assert pd.isna(result.loc[4, "momentum_5d"])
    assert result.loc[5, "momentum_5d"] == pytest.approx(0.25)


def test_add_volatility():
    data = pd.DataFrame(
        {
            "Close": [100, 102, 101, 104, 103, 105],
        }
    )

    data = add_daily_return(data)
    result = add_volatility(data, window=5)

    assert pd.isna(result.loc[0, "volatility_5d"])
    assert pd.isna(result.loc[4, "volatility_5d"])
    assert pd.isna(result.loc[5, "volatility_5d"]) is False
