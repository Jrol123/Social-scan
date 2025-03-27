import pytest
from get_url.get_url import Finder

def test_Finder_Google(q: str, expected: dict[str, str]):
    result = Finder.find(q, "Google")
    assert expected.get("Google") == result.get("Google")
    
def test_Finder_Yandex(q: str, expected: dict[str, str]):
    result = Finder.find(q, "Yandex")
    assert expected.get("Yandex") == result.get("Yandex")