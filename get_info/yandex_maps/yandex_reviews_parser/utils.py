import time
import undetected_chromedriver
from .parsers import Parser
from selenium.webdriver.common.by import By

import logging
logging.basicConfig(
    level=logging.INFO,
    filename="parsing.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding='utf-8'
)

# TODO: Добавить credits автору оригинального репозитория
class YandexParser:

    def __init__(self, id_yandex: int):
        """
        @param id_yandex: ID Яндекс компании
        """
        self.id_yandex = id_yandex

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

    def parse(self, type_parse: str = 'default', sort_type = 'Сначала отрицательные') -> dict:
        """
        Функция получения данных в виде
        @param type_parse: Тип данных, принимает значения:
            default - получает все данные по аккаунту
            company - получает данные по компании
            reviews - получает данные по отчетам
        @return: Данные по запрашиваемому типу
        """
        logging.info("ПРОЦЕСС НАЧАТ")
        result: dict = {}
        driver, page = self.__open_page()
        
        time.sleep(4)  # Задержка для полной загрузки страницы

        try:
            logging.info(f"СОРТИРОВКА ПО '{sort_type}'")
            # Клик на первый div
            self.__click_element(driver, By.CLASS_NAME, 'rating-ranking-view')

            # Клик на второй div
            self.__click_element(driver, By.CLASS_NAME, 'rating-ranking-view__popup-line', 'Сначала отрицательные')
            
            logging.info("СОРТИРОВКА УСПЕШНА")
            
            
            # checker = driver.find_element(By.CLASS_NAME, value)

            # Парсинг данных в зависимости от типа
            if type_parse == 'default':
                result = page.parse_all_data()
            elif type_parse == 'company':
                result = page.parse_company_info()
            elif type_parse == 'reviews':
                result = page.parse_reviews()

        except Exception as e:
            print(e)
            return result
        finally:
            driver.close()
            driver.quit()
            logging.info("ПРОЦЕСС ЗАВЕРШЁН")
            return result
