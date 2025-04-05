import time
import logging

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import undetected_chromedriver

from .__parsing_process import Parser

logging.basicConfig(
    level=logging.INFO,
    filename="parsing.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
# https://2gis.ru/crimea/firm/70000001046404911/tab/reviews


class TGisParser:
    def __init__(self, loc: str, id_tgis: int) -> None:
        self.loc = loc
        self.id_tgis = id_tgis

    def parse(self, sort_type="Отрицательные") -> dict:
        logging.info("ПРОЦЕСС НАЧАТ")
        result: dict = {}
        driver, page = self.__open_page()

        time.sleep(4)  # Задержка для полной загрузки страницы

        try:
            logging.info(f"СОРТИРОВКА ПО '{sort_type}'")
            # Клик на первый div
            self.__click_element(driver, By.XPATH, f".//label[@title='{sort_type}']")
            #! TODO: данный xpath не работает

            # # Клик на второй div
            # self.__click_element(driver, By.CLASS_NAME, 'rating-ranking-view__popup-line', 'Сначала отрицательные')

            logging.info("СОРТИРОВКА УСПЕШНА")

            result = page.parse()

        except Exception as e:
            print(e)
            return result
        finally:
            driver.close()
            driver.quit()
            logging.info("ПРОЦЕСС ЗАВЕРШЁН")
            return result

    def __open_page(self):
        url = f"https://2gis.ru/{self.loc}/firm/{self.id_tgis}/tab/reviews"
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("headless")
        opts.add_argument("--disable-gpu")
        driver = undetected_chromedriver.Chrome(options=opts)
        parser = Parser(driver)
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
            logging.critical(f"Не удалось кликнуть на элемент: {value}", exc_info=True)


# label title="Отрицательные"
# Можно парсить отзывы, не разворачивая их
