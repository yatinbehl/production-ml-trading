import pytest

from src.main import calculate_future_price


def test_calculate_future_price():
    result = calculate_future_price(100, 0.10)
    assert result == pytest.approx(110)

