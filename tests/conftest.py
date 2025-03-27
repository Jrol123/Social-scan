import pytest


@pytest.fixture()
def get_url_fefu():
    return {
    "q" : "Двфу",
    "expected" : {
        "Yandex": "1047571026",
        "Google": '%D0%94%D0%B0%D0%BB%D1%8C%D0%BD%D0%B5%D0%B2%D0%BE%D1%81%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D0%B9+%D1%84%D0%B5%D0%B4%D0%B5%D1%80%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9+%D1%83%D0%BD%D0%B8%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D1%82%D0%B5%D1%82/@43.119189,131.8096523,13z/data=!4m10!1m2!2m1!1z0JTQstGE0YM!3m6!1s0x5fb39209e3e6c0db:0x8bd2648973447e72!8m2!3d43.119189!4d131.88587!15sCgjQlNCy0YTRgyIDiAEBkgEKdW5pdmVyc2l0eeABAA!16zL20vMDY3ajhr?entry=ttu&g_ep=EgoyMDI1MDMyNC4wIKXMDSoASAFQAw%3D%3D',
        #!п TODO: Во время тестов на Github буквы выходят нормальными. Учитывать эту возможность
    }}
