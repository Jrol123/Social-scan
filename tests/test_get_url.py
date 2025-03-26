import pytest
from get_url.get_url import Finder

def test_Finder(q: str, expected: dict[str, str]):
    result = Finder(q)
    assert expected.get("Yandex") == result.get("Yandex")
    assert expected.get("Google") == result.get("Google")