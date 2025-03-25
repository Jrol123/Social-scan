import time
import undetected_chromedriver
from selenium.webdriver.common.by import By

import logging

logging.basicConfig(
    level=logging.INFO,
    filename="finder.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)


class Parser:
    def __init__(self, service_info: dict[str, str], q: str):
        """
        Args:
            service_info (dict[str, str]): Информация о сервисе
            q (str): поисковый запрос
        """
        self.url = service_info.get("url")
        self.input_xpath = service_info.get("input_xpath")
        self.card_xpath = service_info.get("card_xpath")
        self.q = q

    def __open_page(self):
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("headless")
        opts.add_argument("--disable-gpu")
        driver = undetected_chromedriver.Chrome(options=opts)
        driver.get(self.url)
        return driver

    def __input_element(self, driver: undetected_chromedriver.Chrome):
        """
        Ввод информации в поисковую строку
        """
        try:
            input_element = driver.find_element(By.XPATH, self.input_xpath)
            input_element.send_keys(self.q)
            input_element.submit()

            time.sleep(1)
        except:
            logging.critical(f"Не удалось ввести поисковый запрос: {self.q}", exc_info=True)

    def __click_card(self, driver: undetected_chromedriver.Chrome):
        list_cards = driver.find_elements(By.XPATH, self.card_xpath)
        card = list_cards[0]
        card.click()
        
        time.sleep(1)

    def find(self) -> dict:
        logging.info("ПРОЦЕСС НАЧАТ")
        result: str = None
        driver = self.__open_page()

        time.sleep(4)  # Задержка для полной загрузки страницы

        try:
            self.__input_element(driver)

            self.__click_card(driver)
            result = "/".join(driver.current_url.split("/")[-2:-2 + 1])
            # TODO: Вынести -2 в self

        except Exception as e:
            print(e)
            return result
        finally:
            driver.close()
            driver.quit()
            logging.info("ПРОЦЕСС ЗАВЕРШЁН")
            return result


class Finder:
    yandex = {
        "url": "https://yandex.ru/maps/",
        "input_xpath": ".//input[@class='input__control _bold']",
        "card_xpath": ".//li[@class='search-snippet-view']",
    }
    google = {"url": "https://www.google.com/maps/", "xpath": None, "card_xpath": None}

    def __init__(self, meta_name: str):
        """
        Args:
            meta_name (str): Название объекта, по которому будет происходить поиск
        """
        self.meta_name = meta_name
        yandex_finder = Parser(self.yandex, meta_name)
        # google_finder = Parser(self.google, meta_name)
        self.finder_collection = {"Yandex": yandex_finder, 
                                #   "Google": google_finder,
                                  }

    def find(self) -> dict[str, str]:
        """
        Производит поиск по всем сервисам
        """
        result = {}
        for name, finder in self.finder_collection.items():
            result[name] = finder.find()
        return result

    def save_info(self):
        """
        Сохраняет информацию в бд
        """
        pass
