import pandas as pd
import pytest

from src.features import add_daily_return


def test_add_daily_return():
    data = pd.DataFrame(
        {
            "Close": [100, 105, 102],
        }
    )

    result = add_daily_return(data)

    assert pd.isna(result.loc[0, "daily_return"])
    assert result.loc[1, "daily_return"] == pytest.approx(0.05)
    assert result.loc[2, "daily_return"] == pytest.approx(-0.0285714, abs=1e-6)
