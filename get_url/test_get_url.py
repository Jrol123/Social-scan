from get_url import Finder

res = Finder.find("Двфу")
Finder.remove_service("Google")
res2 = Finder.find("МРИЯ")
Finder.add_services("Google")
res3 = Finder.find("Кремль")
Finder.set_active_services("Yandex")
res4 = Finder.find("Шоколад")
print(res, "\n", res2, "\n", res3, "\n", res4)
