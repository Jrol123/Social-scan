import pytest


@pytest.fixture()
def get_url_fefu():
    return {
    "q" : "Двфу",
    "expected" : {
        "Yandex": "1047571026",
        "Google": 'Дальневосточный+федеральный+университет/@43.119189,131.8096523,13z/data=!4m10!1m2!2m1!1z0JTQstGE0YM!3m6!1s0x5fb39209e3e6c0db:0x8bd2648973447e72!8m2!3d43.119189!4d131.88587!15sCgjQlNCy0YTRgyIDiAEBkgEKdW5pdmVyc2l0eeABAA!16zL20vMDY3ajhr?entry=ttu&g_ep=EgoyMDI1MDMyNC4wIKXMDSoASAFQAw=='
        #! TODO: Во время тестов на Github буквы выходят нормальными. Учитывать эту возможность
        #! TODO: Буквы исправил, но Google нашёл новое место! Замечательно!
    }}
# urllib.parse.unquote(a)
