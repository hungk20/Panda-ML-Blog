from unittest.mock import patch

import pytest

from trading import InvalidValueError, should_buy, should_buy_with_exception


def test_should_buy_is_true():
    stock_code = "ABC"
    mock_response = {"value": 5}
    with patch("trading.requests.get") as mock_get_data:
        mock_get_data.return_value.json.return_value = mock_response
        assert should_buy(stock_code)


def test_should_buy_is_false():
    stock_code = "ABC"
    mock_response = {"value": 50}
    with patch("trading.requests.get") as mock_get_data:
        mock_get_data.return_value.json.return_value = mock_response
        assert should_buy(stock_code) is False


def test_should_buy_raise_error_when_different_format():
    stock_code = "ABC"
    mock_response = {"value": "20"}
    with patch("trading.requests.get") as mock_get_data:
        mock_get_data.return_value.json.return_value = mock_response
        with pytest.raises(InvalidValueError) as excinfo:
            should_buy_with_exception(stock_code)
        assert "Price is NOT float" in str(excinfo.value)
