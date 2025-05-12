import pytest
from difflib import SequenceMatcher
from src.get_url.get_url import Finder

FULL_MODE = "full"
YANDEX = "Yandex"
GOOGLE = "Google"

def string_compare(s1, s2):
    res = SequenceMatcher(None, s1, s2).ratio()
    return res

def test_Finder_Google(get_url_fefu):
    q, expected = get_url_fefu.values()
    result = Finder.find(q, GOOGLE)
    print(result.get(GOOGLE), '\n', expected.get(GOOGLE))
    ratio = string_compare(result.get(GOOGLE), expected.get(GOOGLE))
    print(ratio)
    assert ratio >= 0.95
    # Url немного отличается каждый раз
    # TODO: Было бы неплохо совместить обычное сравнение строк и функцию

def test_Finder_Yandex(get_url_fefu):
    q, expected = get_url_fefu.values()
    result = Finder.find(q, YANDEX)
    assert result.get(YANDEX) == expected.get(YANDEX)
    # ID всегда один и тот-же
    
    