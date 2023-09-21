import requests


API_URL = "https://example.com/api/stock_data"


class InvalidValueError(Exception):
    """ Raise when data value is NOT in expected format """
    pass


def get_data_from_api(stock_code):
    response = requests.get(API_URL, params={'stock_code': stock_code})
    return response.json()['value']


def should_buy(stock_code):
    price = get_data_from_api(stock_code)
    if price > 10:
        return False
    return True


def should_buy_with_exception(stock_code):
    price = get_data_from_api(stock_code)
    if not isinstance(price, float):
        raise InvalidValueError("Price is NOT float")
    if price > 10:
        return False
    return True
