import time
import logging
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class Parser:
    XPATH_TO_GLOBAL = ".//*[@id='root']/div/div/div[1]/div[1]/div[3]/div/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div/div/div[2]/div[2]"
    XPATH_TO_SELECTIONS = lambda x, y: XPATH_TO_GLOBAL + f"/div[{y}]"

    def __init__(self, driver):
        self.driver = driver

    def parse(self) -> list[dict]:
        logging.info("НАЧАТО ПОЛУЧЕНИЕ ОТЗЫВОВ")
        reviews = []
        elements = self.driver.find_elements(By.XPATH, self.XPATH_TO_GLOBAL + "/*")[5:]
        if len(elements) > 1:
            try:
                self.__scroll_to_bottom(elements[-1], self.XPATH_TO_GLOBAL + "/*")
            except:
                logging.critical("Ошибка при прокрутке страницы", exc_info=True)
            logging.info("ПРОКРУТКА ЗАВЕРШЕНА")
            elements = self.driver.find_elements(By.XPATH, self.XPATH_TO_GLOBAL + "/*")[
                5:
            ]
            logging.info("ПОЛУЧЕНИЕ СПИСКА ОТЗЫВОВ ЗАВЕРШЕНА")
            logging.debug(f"Всего элементов: {len(elements)}")
            for index, elem in enumerate(elements):
                user_div, photo_div, text_and_response_div = elem.find_elements(
                    By.XPATH, "./*"
                )[:]
                user, date, rating = self.__get_user(user_div)
                text, response = self.__get_text(text_and_response_div)
                reviews.append(
                    {
                        "name": user,
                        "date": date,
                        "stars": rating,
                        "text": text,
                        "answer": response,
                    }
                )
        logging.info("ОБРАБОТКА ОТЗЫВОВ ЗАВЕРШЕНА")

    def __get_user(self, div) -> list[str, int, float]:
        
        def __clean_date(date_str) -> float:
            months = {
                        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
                        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
                        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
                    }
                    
            parts = date_str.split()
            if len(parts) != 3:
                raise ValueError("Неверный формат. Используйте: 'день месяц год'")
            
            day_str, month_str, year_str = parts
            
            try:
                day = int(day_str)
                year = int(year_str)
            except ValueError:
                raise ValueError("День и год должны быть числами")
            
            month = months.get(month_str.lower())
            if not month:
                raise ValueError(f"Неизвестный месяц: {month_str}")
            
            try:
                dt = datetime(year, month, day)
            except ValueError as e:
                raise ValueError(f"Ошибка в дате: {e}")
            
            return int(dt.timestamp())
        
        user_date, rating = div.find_elements(By.XPATH, "./div/div/*")[1:]
        user = user_date.find_element(By.XPATH, "./span/span[1]/span").text
        date = __clean_date(user_date.find_element(By.XPATH, "./div").text.split(",")[0])
        

    def __get_text(self, div) -> list[str, str | None]:
        pass

    def __scroll_to_bottom(self, elem, path) -> None:
        """
        Скроллим список до последнего отзыва
        :param elem: Последний отзыв в списке
        :param driver: Драйвер undetected_chromedriver
        :return: None
        """
        self.driver.execute_script("arguments[0].scrollIntoView();", elem)
        time.sleep(1)
        checker = self.driver.find_elements(By.XPATH, path)
        new_elem = checker[-1]
        prb_bttn = new_elem.find_elements(By.XPATH, ".//button")
        if len(prb_bttn) != 0 and prb_bttn[0].text == "Загрузить ещё":
            bttn = prb_bttn[0]
            bttn.click()
            time.sleep(1)
            #! TODO: Иногда не работает.
            # Переделать под убирание кнопки "полезно"
            self.__scroll_to_bottom(checker[-2], path)
        if elem == new_elem:
            return
        self.__scroll_to_bottom(new_elem, path)
