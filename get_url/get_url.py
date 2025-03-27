"""
Файл, лол
"""
import time
import undetected_chromedriver
from selenium.webdriver.common.by import By

import logging

FULL_MODE = "full"
YANDEX = "Yandex"
GOOGLE = "Google"

"""Константы режимов"""

MODES: dict[str, dict[str, str | tuple[int, int]]] = {
    YANDEX: {
        "service_name": "Yandex",
        "url": "https://yandex.ru/maps/",
        "input_xpath": ".//input[@class='input__control _bold']",
        "confirm_xpath": ".//button[@class='button _view_search _size_medium']",
        "card_xpath": ".//li[@class='search-snippet-view']",
        "url_pos": (-2, -2),
    },
    GOOGLE: {
        "service_name": "Google",
        "url": "https://www.google.com/maps/",
        "input_xpath": ".//input",
        "confirm_xpath": ".//button[@id='searchbox-searchbutton']",
        "card_xpath": ".//a",
        "url_pos": (-3, -1),
        #! TODO: у Google постоянно меняются xpath. Возможно, стоит брать по типу (сейчас нужно брать третий <a> и первый инпут)
    },
    # TODO: Попробовать поработать с google.com/maps/search и с его аналогом у Yandex
}
"""Режимы"""

logging.basicConfig(
    level=logging.INFO,
    filename="finder.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)


# TODO: Сделать супер-класс
class Parser:
    def __init__(self, service_info: dict[str, str]):
        """
        Args:
            service_info (dict[str, str]): Информация о сервисе
        """
        self.service_name = service_info.get("service_name")
        self.url = service_info.get("url")
        self.input_xpath = service_info.get("input_xpath")
        self.confirm_xpath = service_info.get("confirm_xpath")
        self.card_xpath = service_info.get("card_xpath")
        self.url_pos = service_info.get("url_pos")

    def __open_page(self):
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("headless")
        opts.add_argument("--disable-gpu")
        driver = undetected_chromedriver.Chrome(options=opts)
        driver.get(self.url)
        return driver

    def __input_element(self, q: str, driver: undetected_chromedriver.Chrome):
        """
        Ввод информации в поисковую строку
        """
        try:
            input_element = driver.find_element(By.XPATH, self.input_xpath)
            input_element.send_keys(q)
            # input_element.submit()
            confirm_element = driver.find_element(By.XPATH, self.confirm_xpath)
            confirm_element.click()
            # Я НЕ ПОНИМАЮ, ПОЧЕМУ ОНО ПЕРИОДИЧЕСКИ НЕ РАБОТАЕТ!!!
            # Скорее всего, связано с интернетом

            time.sleep(1)
        except:
            logging.critical(f"Не удалось ввести поисковый запрос", exc_info=True)

    def __click_card(self, driver: undetected_chromedriver.Chrome):
        try:
            list_cards = driver.find_elements(By.XPATH, self.card_xpath)
            if self.service_name == "Google":
                list_cards = list_cards[2:]
                # Костыль
            card = list_cards[0]
            card.click()
        except:
            if self.service_name == "Google":
                logging.info(f"{self.service_name} автоматически перешёл на карточку объекта")
            else:
                logging.critical(f"Не удалось нажать на карточку", exc_info=True)
        finally:
            time.sleep(1)

    def find(self, q: str) -> str:
        """Поиск по сервису

        Args:
            q (str): поисковый запрос

        Returns:
            str: Результат запроса, который требуется для однозначного определения объекта на сервисе
        """
        logging.info(f"ПРОЦЕСС ДЛЯ {self.service_name} НАЧАТ")
        result: str = None
        driver = self.__open_page()

        time.sleep(4)  # Задержка для полной загрузки страницы

        try:
            self.__input_element(q, driver)
            logging.info(f"ПОИСК ПРОВЕДЁН")

            self.__click_card(driver)
            logging.info(f"ПОИСК КАРТОЧКИ ЗАВЕРШЁН")
            result = "/".join(driver.current_url.split("/")[self.url_pos[0] : self.url_pos[1] + 1 if self.url_pos[1] != -1 else None])

        except:
            logging.critical("ПРОИЗОШЛА ОШИБКА", exc_info=True)
            return result
        finally:
            driver.close()
            driver.quit()
            logging.info(f"ПРОЦЕСС ДЛЯ {self.service_name} ЗАВЕРШЁН")
            return result


class __Finder:
    def __init__(self, mode: str | list[str] = FULL_MODE):
        """
        Args:
            mode (str|list[str]): Список сервисов, по которым будет производиться поиск
        """

        if isinstance(mode, str):
            if mode == FULL_MODE:
                self.finder_collection = {name : Parser(sub_mode) for name, sub_mode in MODES.items()}
                return
            self.finder_collection = {mode : Parser(self.__match_mode(mode))}
        elif isinstance(mode, list):
            self.finder_collection = {sub_mode: Parser(self.__match_mode(sub_mode)) for sub_mode in mode}
        else:
            raise ValueError("Wrong mode")

    def __match_mode(self, mode: str):
        try:
            MODES.get(mode)
        except:
            raise ValueError("Wrong type of site")

    def find(self, q: str, mode: str | list[str] = FULL_MODE) -> dict[str, str]:
        """Производит поиск по всем сервисам
        
        Args:
            q (str): поисковый запрос. Defaults to FULL_MODE.
            
        Returns:
            dict[str, str]: Сервис - результат
        
        """
        logging.info("ПОИСК НАЧАЛСЯ")
        result: dict[str, str] = {}
        if isinstance(mode, str):
            if mode == FULL_MODE:
                for name, finder in self.finder_collection.items():
                    result[name] = finder.find(q)
            else:
                try:
                    result[mode] = self.finder_collection.get(mode).find(q)
                except:
                    raise ValueError(f"Wrong mode {mode}")
        elif isinstance(mode, list):
            for sub_mode in mode:
                try:
                    result[sub_mode] = self.finder_collection.get(sub_mode).find(q)
                except:
                    raise ValueError(f"Wrong mode {sub_mode}")
        else:
            raise ValueError(f"Wrong mode {mode}")
        logging.info("ПОИСК ЗАВЕРШЁН")
        
        return result

    def save_info(self):
        """
        Сохраняет информацию в бд
        """
        pass
    
Finder = __Finder(FULL_MODE)

