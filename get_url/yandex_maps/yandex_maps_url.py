import time
import undetected_chromedriver
from selenium.webdriver.common.by import By

import logging
logging.basicConfig(
    level=logging.INFO,
    filename="finder.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding='utf-8'
)

class Parser:
    def __init__(self, q):
        """
        @param id_yandex: поисковое значение
        """
        self.q

    def __open_page(self):
        url: str = 'https://yandex.ru/maps/org/{}/reviews/'.format(str(self.id_yandex))
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('headless')
        opts.add_argument('--disable-gpu')
        driver = undetected_chromedriver.Chrome(options=opts)
        parser = Parser(driver)
        driver.get(url)
        return driver, parser

    def __click_element(self, driver, by, value, find_value = None):
        """ Функция для клика на элемент с ожиданием. """
        try:
            # Ожидание 10 секунд, пока элемент не станет кликабельным
            elements = driver.find_elements(by, value)
            element = elements[0]
            
            # TODO: Костыль, т. к. не имею понятия, почему не работает xpath по нескольким критериям
            if len(elements) > 1:
                for el in elements:
                    if el.text == find_value:
                        element = el
                        break
            
            element.click()
            time.sleep(1)  # Небольшая задержка для отработки клика
        except Exception as e:
            # TODO: Почему-то периодически вылезает "неудалось кликнуть"
            logging.critical(f"Не удалось кликнуть на элемент: {value}", exc_info=True)
class Finder:
    yandex_find = 'https://yandex.ru/maps/'
    google_find = 'https://www.google.com/maps/'
    def __init__(self, meta_name:str):
        """
        Args:
            meta_name (str): Название объекта, по которому будет происходить поиск
        """
        self.meta_name = meta_name
        
    def save_info(self):
        """Сохраняет информацию в бд
        """
        pass
    
    