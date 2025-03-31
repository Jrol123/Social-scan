"""
Файл, лол
"""
import time
import undetected_chromedriver
from selenium.webdriver.common.by import By
from sqlite3 import connect
import json

import logging

FULL_MODE = "full"
"""Работа со всеми режимами"""

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

            time.sleep(4)
        except:
            logging.critical(f"Не удалось ввести поисковый запрос", exc_info=True)

    def __click_card(self, driver: undetected_chromedriver.Chrome):
        """
        Нажатие на карточку
        """
        try:
            list_cards = driver.find_elements(By.XPATH, self.card_xpath)
            if self.service_name == GOOGLE:
                list_cards = list_cards[2:]
                # Костыль
            card = list_cards[0]
            card.click()
        except:
            if self.service_name == GOOGLE:
                logging.info(f"{self.service_name} автоматически перешёл на карточку объекта")
            else:
                logging.critical(f"Не удалось нажать на карточку", exc_info=True)
        finally:
            time.sleep(4)

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

# TODO: Переделать константы под получение информации из базы данных
class __Finder:
    def __init__(self, service_names: str | list[str] = FULL_MODE):
        """
        Args:
            service_names (str|list[str]): Список сервисов, по которым будет производиться поиск. Defaults to FULL_MODE.
        """
        
        self.services = self.__load_services()
        self.active_services = self.set_active_services(service_names)
        
    def set_active_services(self, service_names: str | list[str] = FULL_MODE):
        """
        Задание активных сервисов

        Args:
            service_names (str | list[str]): Список сервисов, по которым будет производиться поиск. Defaults to FULL_MODE.

        Raises:
            TypeError: Неправильный тип данных
        """
        if isinstance(service_names, str):
            if service_names == FULL_MODE:
                self.active_services = self.services.copy()
            else:
                self.active_services = {service_names : Parser(self.__match_mode(service_names))}
        elif isinstance(service_names, list):
            self.active_services = {name : Parser(self.services[name]) for name in service_names}
        else:
            raise TypeError(f"Неправильный тип для {service_names}: {type(service_names)}")

    def find(self, q: str, mode: str | list[str] = FULL_MODE) -> dict[str, str]:
        """Производит поиск по всем сервисам
        
        Args:
            q (str): поисковый запрос.
            mode (str|list[str]): Сервисы, с которыми будет происходить работа. Defaults to FULL_MODE.
            
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
                    raise ValueError(f"Сервис {mode} не найден в БД")
        elif isinstance(mode, list):
            for sub_mode in mode:
                try:
                    result[sub_mode] = self.finder_collection.get(sub_mode).find(q)
                except:
                    raise ValueError(f"Сервис {sub_mode} не найден в БД")
        else:
            raise TypeError(f"Неправильный тип для {service_names}: {type(service_names)}")

        logging.info("ПОИСК ЗАВЕРШЁН")

        return result

    def save_info(self):
        """
        Сохраняет информацию в бд
        """
        pass

    def __match_mode(self, service_names: str):
        for name in service_names:
            if name not in self.services:
                raise ValueError(f"Сервис {name} не найден в БД")
        self.active_services = {name: self.services[name] for name in service_names}

    def __load_services(self) -> dict[str, dict]:
        """
        Загружает сервисы из БД
        """
        with connect("main.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT service_name, parsing_config FROM Services")
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}


Finder = __Finder()
