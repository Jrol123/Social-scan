import time
import logging

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class Parser:
    XPATH_TO_GLOBAL = ".//*[@id='root']/div/div/div[1]/div[1]/div[3]/div/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div/div/div[2]/div[2]"
    XPATH_TO_SELECTIONS = lambda x, y: XPATH_TO_GLOBAL + f"/div[{y}]"

    def __init__(self, driver):
        self.driver = driver

    def parse(self) -> dict:
        logging.info("НАЧАТО ПОЛУЧЕНИЕ ОТЗЫВОВ")
        reviews = []
        elements = self.driver.find_elements(By.XPATH, self.XPATH_TO_GLOBAL + "/*")[5:]
        if len(elements) > 1:
            try:
                self.__scroll_to_bottom(elements[-1], self.XPATH_TO_GLOBAL + "/*")
            except:
                logging.critical("Ошибка при прокрутке страницы", exc_info=True)
            logging.info("ПРОКРУТКА ЗАВЕРШЕНА")
            elements = self.driver.find_elements(By.XPATH, self.XPATH_TO_GLOBAL + "/*")[5:]
            logging.info("ПОЛУЧЕНИЕ СПИСКА ОТЗЫВОВ ЗАВЕРШЕНА")
            logging.debug(f"Всего элементов: {len(elements)}")
            for index, elem in enumerate(elements):
                user_div, photo_div, text_and_response_div = elem.find_elements(By.XPATH, "./*")[:]
                pass
                # reviews.append(self.__get_data_item(index, elem))
        logging.info("ОБРАБОТКА ОТЗЫВОВ ЗАВЕРШЕНА")
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
