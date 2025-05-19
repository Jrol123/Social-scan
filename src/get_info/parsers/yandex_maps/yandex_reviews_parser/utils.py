import logging
import time

import undetected_chromedriver
from selenium.webdriver.common.by import By

from .config import YandexMapsConfig
from .parser import Parser
from ....abstract import Parser as aParser
from ....core import MasterParserConfig


logging.basicConfig(
    level=logging.INFO,
    filename="parsing.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)


# TODO: Добавить credits автору оригинального репозитория
class YandexMapsParser(aParser):

    SORT_DICT = {
        "rating_ascending": "Сначала отрицательные",
        "rating_descending": "Сначала положительные",
        "date_descending": "По новизне",
        "default": "По умолчанию",
    }

    def __init__(self, local_config: YandexMapsConfig):
        super().__init__(1, local_config)  # TODO: Считывать id из .txt

    def __open_page(self, id_yandex: int):
        url: str = f"https://yandex.ru/maps/org/{id_yandex}/reviews/"
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("headless")
        opts.add_argument("--disable-gpu")
        driver = undetected_chromedriver.Chrome(options=opts)
        parser = Parser(driver, self.service_id)
        driver.get(url)
        return driver, parser

    def __click_element(self, driver, by, value, find_value=None):
        """Функция для клика на элемент с ожиданием."""
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

    def parse(self, global_config: MasterParserConfig) -> dict:
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

        min_date = self._date_convert(global_config.min_date, int)
        max_date = self._date_convert(global_config.max_date, int)

        sort_type = self.SORT_DICT[global_config.sort_type]

        driver, page = self.__open_page(self.config.q)

        time.sleep(4)  # Задержка для полной загрузки страницы

        try:
            logging.info(f"СОРТИРОВКА ПО '{sort_type}'")
            # Клик на первый div
            self.__click_element(driver, By.CLASS_NAME, "rating-ranking-view")

            # Клик на второй div
            self.__click_element(
                driver,
                By.CLASS_NAME,
                "rating-ranking-view__popup-line",
                sort_type,
            )

            logging.info("СОРТИРОВКА УСПЕШНА")

            # checker = driver.find_element(By.CLASS_NAME, value)

            # Парсинг данных в зависимости от типа
            result = page.parse_reviews(min_date, max_date, global_config.count_items)[
                "company_reviews"
            ]

        except Exception as e:
            print(e)
            return result
        finally:
            driver.close()
            driver.quit()
            logging.info("ПРОЦЕСС ЗАВЕРШЁН")
            return result
