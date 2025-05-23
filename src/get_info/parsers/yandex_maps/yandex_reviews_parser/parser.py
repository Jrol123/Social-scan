import time
import logging
from dataclasses import asdict

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from .helpers import ParserHelper
from .storage import Review, Info

# NOTICE_LEVEL = 25
# logging.addLevelName(NOTICE_LEVEL, "NOTICE")

# def notice(self, message, *args, **kws):
#     if self.isEnabledFor(NOTICE_LEVEL):
#         self._log(NOTICE_LEVEL, message, args, **kws)

# logging.Logger.notice = notice

logging.basicConfig(
    level=logging.INFO,
    filename="parsing.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    encoding='utf-8'
)

class Parser:
    def __init__(self, driver, service_id):
        self.driver = driver
        self.service_id = service_id

    def __scroll_to_bottom(self, elem) -> None:
        """
        Скроллим список до последнего отзыва
        :param elem: Последний отзыв в списке
        :param driver: Драйвер undetected_chromedriver
        :return: None
        """
        self.driver.execute_script(
            "arguments[0].scrollIntoView();",
            elem
        )
        time.sleep(1)
        checker = self.driver.find_elements(By.CLASS_NAME, "business-reviews-card-view__review")
        new_elem = checker[-1]
        if elem == new_elem:
            return
        self.__scroll_to_bottom(new_elem)

    def __get_data_item(self, index, elem, min_date, max_date):
        """
        Спарсить данные по отзыву
        :param elem: Отзыв из списка
        :return: Словарь
        {
            name: str
            icon_href: Union[str, None]
            date: int
            text: str
            rating: float
        }
        """
        try:
            name = elem.find_element(By.XPATH, ".//span[@itemprop='name']").text
        except NoSuchElementException:
            logging.warning(f"{index}. Не найден блок с именем пользователя", exc_info=True)
            name = None

        # try:
        #     icon_href = elem.find_element(By.XPATH, ".//div[@class='user-icon-view__icon']").get_attribute('style')
        #     icon_href = icon_href.split('"')[1]
        # except NoSuchElementException:
        #     logging.debug(f"{index}. Не найден блок с иконкой пользователя")  # При отсутствии не находится
        #     icon_href = None

        try:
            date = elem.find_element(By.XPATH, ".//meta[@itemprop='datePublished']").get_attribute('content')
        except NoSuchElementException:
            logging.warning(f"{index}. Не найден блок с датой публикации", exc_info=True)
            date = None
        
        date=ParserHelper.form_date(date)
        
        if not (date <= max_date):
            return 1
        
        if not (min_date <= date):
            return -1

        try:
            text = elem.find_element(By.XPATH, ".//span[@class='business-review-view__body-text']").text
        except NoSuchElementException:
            logging.warning(f"{index}. Не найден блок с текстом отзыва", exc_info=True)
            text = None
            
        if text is None or "":
            return 1
        
        try:
            rating = elem.find_element(By.XPATH, ".//meta[@itemprop='ratingValue']").get_attribute('content')
        except NoSuchElementException:
            logging.warning(f"{index}. Не найден блок с оценкой", exc_info=True)
            rating = 0

        try:
            answer = elem.find_element(By.CLASS_NAME, "business-review-view__comment-expand")
            if answer:
                self.driver.execute_script("arguments[0].click()", answer)
                answer = elem.find_element(By.CLASS_NAME, "business-review-comment-content__bubble").text
            else:
                answer = None
        except NoSuchElementException:
            logging.debug(f"{index}. Не найден блок с ответом объекта")  # При отсутствии не находится
            answer = None
        item = Review(
            service_id=self.service_id,
            name=name,
            # icon_href=icon_href,
            additional_id=None,
            date=date,
            text=text,
            rating=rating,
            answer=answer
        )
        return asdict(item)

    def __get_data_campaign(self) -> dict:
        """
        Получаем данные по компании.
        :return: Словарь данных
        {
            name: str
            rating: float
            count_rating: int
            stars: float
        }
        """
        logging.info("НАЧАТО ПОЛУЧЕНИЕ ОБЪЕКТА")
        try:
            xpath_name = ".//h1[@class='orgpage-header-view__header']"
            name = self.driver.find_element(By.XPATH, xpath_name).text
        except NoSuchElementException:
            logging.critical("Заголовок объекта не найден", exc_info=True)
            name = None
        try:
            xpath_rating_block = ".//div[@class='business-summary-rating-badge-view__rating-and-stars']"
            rating_block = self.driver.find_element(By.XPATH, xpath_rating_block)
            try:
                xpath_count_rating = ".//div[@class='business-summary-rating-badge-view__rating-count']/span[@class='business-rating-amount-view _summary']"
                count_rating_list = rating_block.find_element(By.XPATH, xpath_count_rating).text
                count_rating = ParserHelper.list_to_num(count_rating_list)
            except NoSuchElementException:
                logging.warning("Не найден блок с количеством отзывов", exc_info=True)
                count_rating = 0
            try:
                xpath_stars = ".//div[@class='business-summary-rating-badge-view__rating']"
                stars_text = rating_block.find_element(By.XPATH, xpath_stars).text
                stars_text_split = stars_text.split('\n')[-1].replace(',', '.')
                # TODO: Переделать на нормальный элемент, без split-а.
                stars = float(stars_text_split)
            except NoSuchElementException:
                logging.warning("Не найден блок с оценкой", exc_info=True)
                stars = 0
            except ValueError:
                logging.warning("Не удалось перевести оценку в число", exc_info=True)
                stars = 0
        except NoSuchElementException:
            logging.error("Подзаголовок с информацией не найден", exc_info=True)
            count_rating = 0
            stars = 0

        item = Info(
            name=name,
            rating=count_rating,
            stars=stars
        )
        return asdict(item)

    def __get_data_reviews(self, min_date, max_date, count_items) -> list:
        logging.info("НАЧАТО ПОЛУЧЕНИЕ ОТЗЫВОВ")
        reviews = []
        elements = self.driver.find_elements(By.CLASS_NAME, "business-reviews-card-view__review")
        if len(elements) > 1:
            self.__scroll_to_bottom(elements[-1])
            logging.info("ПРОКРУТКА ЗАВЕРШЕНА")
            elements = self.driver.find_elements(By.CLASS_NAME, "business-reviews-card-view__review")
            logging.info("ПОЛУЧЕНИЕ СПИСКА ОТЗЫВОВ ЗАВЕРШЕНА")
            logging.debug(f"Всего элементов: {len(elements)}")
            for index, elem in enumerate(elements):
                res_parse = self.__get_data_item(index, elem, min_date, max_date)
                if res_parse == -1:
                    break
                if res_parse == 1:
                    continue
                reviews.append(res_parse)
                if count_items != -1 and len(reviews) == count_items:
                    break
            logging.info("ОБРАБОТКА ОТЗЫВОВ ЗАВЕРШЕНА")
        return reviews

    def __isinstance_page(self):
        try:
            xpath_name = ".//h1[@class='orgpage-header-view__header']"
            name = self.driver.find_element(By.XPATH, xpath_name).text
            logging.info(f"Заголовок объекта {name} найден")
            return True
        except NoSuchElementException:
            logging.critical("Заголовок объекта не найден", exc_info=True)
            return False

    def parse_all_data(self, min_date, max_date, count_items) -> dict:
        """
        Начинаем парсить данные.
        :return: Словарь данных
        {
             company_info:{
                    name: str
                    count_rating: int
                    rating: float
            },
            company_reviews:[
                {
                  name: str
                  icon_href: str
                  date: timestamp
                  text: str
                  rating: float
                }
            ]
        }
        """
        if not self.__isinstance_page():
            return {'error': 'Страница не найдена'}
        data_campaign = self.__get_data_campaign()
        logging.info("ИНФОРМАЦИЯ О КОМПАНИИ ПОЛУЧЕНА УСПЕШНО")
        data_reviews = self.__get_data_reviews(min_date, max_date, count_items)
        logging.info("ИНФОРМАЦИЯ ОБ ОТЗЫВАХ ПОЛУЧЕНА УСПЕШНО")
        return {'company_info': data_campaign, 'company_reviews': data_reviews}

    def parse_reviews(self, min_date, max_date, count_items) -> dict:
        """
        Начинаем парсить данные только отзывы.
        :return: Массив отзывов
        {
            company_reviews:[
                {
                  name: str
                  icon_href: str
                  date: timestamp
                  text: str
                  stars: float
                }
            ]
        }

        """
        if not self.__isinstance_page():
            return {'error': 'Страница не найдена'}
        return {'company_reviews': self.__get_data_reviews(min_date, max_date, count_items)}

    def parse_company_info(self) -> dict:
        """
        Начинаем парсить данные только данные о компании.
        :return: Объект компании
        {
            company_info:
                {
                    name: str
                    count_rating: int
                    stars: float
                }
        }
        """
        if not self.__isinstance_page():
            return {'error': 'Страница не найдена'}
        return {'company_info': self.__get_data_campaign()}
