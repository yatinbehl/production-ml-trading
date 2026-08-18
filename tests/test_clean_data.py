import pandas as pd

from src.clean_data import clean_columns


def test_clean_columns():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Open", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )

    data = pd.DataFrame(
        [[100, 105, 95, 98, 1000000]],
        columns=columns,
    )

    result = clean_columns(data)

    assert result.columns.tolist() == [
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    ]
