import logging
import re
import time
from datetime import datetime, timedelta

import emoji
import pandas as pd
import undetected_chromedriver
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from .config import GoogleMapsConfig
from ...abstract import Parser
from ...core import MasterParserConfig


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


SORT_DICT = {
        "rating_ascending": "Сначала отрицательные",
        "rating_descending": "Сначала положительные",
        "date_descending": "По новизне",
        "default": "По умолчанию",
    }

SORT_TYPES = {'default': 'Самые релевантные',
              'date_descending': 'Сначала новые',
              'rating_ascending': 'По возрастанию рейтинга',
              'rating_descending': 'По убыванию рейтинга',}
"""
Возможные виды сортировок.
"""


class GoogleMapsParser(Parser):
    def __init__(self, config: GoogleMapsConfig):
        super().__init__(0, config)
        
    
    time_units = {'вчера': timedelta(days=1), 'день': timedelta(days=1),
                  'дн': timedelta(days=1), 'недел': timedelta(weeks=1)}
    
    def parse(
        self,
        global_config: MasterParserConfig
    ) -> list[dict[str, str | int | float | None]]:
        sort_type = global_config.sort_type
        assert sort_type in SORT_DICT, "Нет такого вида сортировки"
        
        count_items = global_config.count_items
        min_date = self._date_convert(global_config.min_date, datetime)
        max_date = self._date_convert(global_config.max_date, datetime)
        
        driver = self.__initialize_browser(self.config.q)
        reviews = []
        # try:
        # Wait for the business details to load
        time.sleep(3)
        
        # # Locate and click the reviews section
        # # logger.info("Searching for reviews section")
        self.__click_element(driver, By.XPATH, "//div[contains(text(), 'Отзывы')]")
        time.sleep(1)
        
        # Choose google reviews
        if driver.find_elements(By.XPATH,
                                "//div[contains(text(), 'Все отзывы')]"):
            self.__click_element(driver, By.XPATH,
                                 "//div[contains(text(), 'Все отзывы')]")
            time.sleep(0.5)
            self.__click_element(driver, By.XPATH,
                                 f"//div[contains(text(), 'Google')]")
            time.sleep(3)
        
        # Choose sorting type
        if sort_type != 'default':
            self.__click_element(driver, By.XPATH,
                                 "//div[contains(text(), 'Самые релевантные')]")
            time.sleep(0.5)
            self.__click_element(driver, By.XPATH,
                                 f"//div[contains(text(), '{SORT_TYPES[sort_type]}')]")
            time.sleep(3)
        
        # Scroll to load more reviews
        # logger.info("Loading reviews...")
        
        now = datetime.now()
        # time.sleep(30)
        try:
            prev_element = driver.find_elements(By.CSS_SELECTOR,
                                                "div[data-review-id] > div")[-1]
        except IndexError:
            print(driver.find_elements(By.CSS_SELECTOR,
                                                "div[data-review-id] > div"))
            raise
        
        last_checked = 0
        continue_parsing = True
        while continue_parsing:
            self.__expand_reviews(driver)
            # self.__scroll_reviews(driver)
            # time.sleep(3)
            curr_element = driver.find_elements(
                By.CSS_SELECTOR,"div[data-review-id] > div")[-1]
            # print(curr_element)
            
            time.sleep(2)
            if prev_element == curr_element:
                # Waiting for next reviews to load
                for _ in range(self.config.wait_load):
                    time.sleep(1)
                    curr_element = driver.find_elements(
                        By.CSS_SELECTOR, "div[data-review-id] > div")[-1]
                    logging.debug(f"Waiting load time: {self.config.wait_load}")
                    if prev_element != curr_element:
                        break
                else:  # If time is expired, consider we reached the bottom
                    print(last_checked)
                    print(curr_element.text)
                    break
            
            prev_element = curr_element
            
            # self.__expand_reviews(driver)
            self.__scroll_reviews(driver)

            # Extract reviews
            # logger.info(f"Found {review_elements.count()} reviews")
            
            review_elements = driver.find_elements(
                By.CSS_SELECTOR, "div[data-review-id] > div"
            )
            print(len(review_elements), last_checked)
            for element in review_elements[last_checked + 1::2]:
                reviewer = element.find_element(
                    By.CSS_SELECTOR,
                    "div:nth-child(2) > div"
                    " > button[data-review-id]:first-child > div:first-child").text
                
                rating = element.find_element(
                    By.CSS_SELECTOR,
                    "div:nth-child(4) > div:first-child > span:first-child")
                if rating.get_attribute('role'):
                    rating = int(rating.get_attribute('aria-label')[0])
                else:
                    rating = int(rating.text[0])
                
                date = element.find_element(
                    By.CSS_SELECTOR,
                    "div:nth-child(4) > div:first-child > span:nth-child(2)").text
                date = date.rsplit(',', 1)[0].strip()
                
                # Prepare data for output
                date = self.text_to_date(date.lower(), now)
                if (sort_type == 'date_descending' and
                   min_date is not None and date < min_date):
                    continue_parsing = False
                    break
                elif (sort_type != 'date_descending'
                      and ((min_date is not None and date < min_date)
                           or (max_date is not None and date > max_date))):
                    continue
                else:
                    date = int(date.timestamp())
                
                if self.config.collect_extra:
                    text_selector = "div > div > div[tabindex='-1'][id]"
                else:
                    text_selector = ("div > div > div[tabindex='-1'][id]"
                                     " > span:first-child")
                
                try:
                    review_text = element.find_element(
                        By.CSS_SELECTOR, text_selector).text
                    review_text = review_text.replace('\n', ' ').replace('\t', ' ')
                except Exception:
                    continue
                
                answer = element.find_elements(By.CSS_SELECTOR,
                                               "div:nth-child(4) > div")[-1]
                subtitle = answer.find_elements(
                    By.CSS_SELECTOR, "div:first-child > span:first-child")
                if answer and subtitle and "Ответ владельца" in subtitle[-1].text:
                    answer = answer.find_element(By.CSS_SELECTOR,
                                                 "div:nth-child(2)").text
                    answer = answer.replace('\n', ' ').replace('\t', ' ')
                else:
                    answer = None
                    
                reviews.append({
                    "service_id": self.service_id,
                    "name": self.__clean_text(reviewer),
                    "additional_id": None,
                    "date": date,
                    "rating": rating,
                    "text": self.__clean_text(review_text),
                    "answer": self.__clean_text(answer)
                })
                
            if count_items != -1 and len(reviews) >= count_items:
                continue_parsing = False
                break
            # else:
            #     print(len(reviews))
            
            last_checked = len(review_elements)
        
        # except Exception as e:
        #     raise e
            # logger.error(f"Error during scraping: {e}")
        # finally:
        time.sleep(5)
        driver.quit()
        
        if count_items != -1:
            reviews = reviews[:count_items]
            
        return reviews
        
    @staticmethod
    def get_total_reviews(driver):
        # Получение общего числа отзывов
        total_reviews = driver.find_element(
            By.XPATH, "//div[contains(text(), 'Отзывов:')]").text
        total_reviews = int(total_reviews.split(': ', 1)[1].replace(' ', '')
                            .replace('&nbsp;', '').strip())
        return total_reviews
    
    def text_to_date(self, text, now):
        date = text.rsplit(' ', 1)[0]
        if date[0].isdigit():
            try:
                num, unit = date.split(' ')
                num = int(num)
            except ValueError:
                i = 2 if date[1].isdigit() else 1
                num, unit = date[:i], date[i + 1:]
                num = int(num)
        else:
            unit = date
            num = 1
        
        if unit.startswith('год') or unit == 'лет':
            date = now.replace(year=now.year - num)
        elif unit.startswith('месяц'):
            date = now.replace(year=now.year if now.month > num else now.year - 1,
                               month=now.month - num if now.month > num
                               else now.month - num + 12)
        else:
            for key in self.time_units:
                if unit.startswith(key):
                    unit = key
                    break
            
            date = now - num * self.time_units[unit]
        
        return date
    
    @staticmethod
    def __initialize_browser(url):
        # driver = webdriver.Chrome()
        opts = undetected_chromedriver.ChromeOptions()
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('headless')
        opts.add_argument('--disable-gpu')
        driver = undetected_chromedriver.Chrome(options=opts)
        driver.get(url if url.startswith('https')
                   else "https://www.google.com/maps/place/" + url)
        # time.sleep(2)
        return driver

    @staticmethod
    def __clean_text(text):
        if not text:
            return text
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove extra whitespace
        text = re.sub(r's+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def __click_element(driver, by=By.CSS_SELECTOR, value=None, find_value=None):
        elements = driver.find_elements(by, value)
        if len(elements) == 0:
            return
        
        for i, elem in enumerate(elements):
            if find_value and find_value in elem.text:
                continue
                
            try:
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(elem)
                )
                break
            except Exception as e:
                print(f"Элемент {i + 1} не кликабельный или возникла ошибка: {e}")
        
        try:
            element.click()
        except Exception:
            driver.execute_script("arguments[0].click();", element)

    @staticmethod
    def __scroll_reviews(driver, wait_sec=2):
        last_element = driver.find_elements(
            By.CSS_SELECTOR,
            "div[role='main'] > div > div > div[aria-hidden='true']:last-child")[-1]
        driver.execute_script("arguments[0].scrollIntoView();", last_element)
        time.sleep(wait_sec)

    @staticmethod
    def __expand_reviews(driver):
        try:
            # Trying to load buttons for review texts expenditure
            load_more_buttons = WebDriverWait(driver, 30).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, "//button[contains(text(), 'Ещё')]"))
            )
            if not load_more_buttons:
                print("Нет кнопок 'Ещё' для загрузки.")
                return
            
            # Go to the first button on page
            driver.execute_script("arguments[0].scrollIntoView();",
                                  load_more_buttons[0])
            time.sleep(2)
            for button in load_more_buttons:
                driver.execute_script("arguments[0].scrollIntoView();", button)
                
                # Check if button is clickable
                if button.is_displayed() and button.is_enabled():
                    try:
                        button.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", button)
            else:
                # Если цикл завершился без нахождения кнопки для нажатия
                print("Больше нет кнопок 'Ещё' для загрузки.")
                return
        
        except Exception as e:
            print("Произошла ошибка:", e)
            return
    
    def __check_date(self, curr_element, min_date, now):
        curr_date = curr_element.find_element(
            By.CSS_SELECTOR,
            "div:nth-child(4) > div:first-child > span:nth-child(2)"
        ).text
        curr_date = curr_date.rsplit(',', 1)[0].strip()
        curr_date = self.text_to_date(curr_date, now)
        return curr_date < min_date


def save_reviews_to_csv(reviews, filename="google_reviews.csv"):
    if not reviews:
        print('С google maps не собрано данных.')
        return
    
    df = pd.DataFrame(reviews)
    try:
        if df.empty:
            return
    except Exception as e:
        raise e

    df.to_csv(filename, index=False, encoding='utf-8')
    # logger.info(f"Reviews saved to {filename}")

def google_maps_parse(q, count_items=100, sorting='ascending', collect_extra=True,
                      min_date=None, max_date=datetime.now(), file="google_reviews.csv"):
    parser = GoogleMapsParser()
    reviews = parser.parse(q, count_items, sorting, min_date, max_date, collect_extra)
    save_reviews_to_csv(reviews, file)


if __name__ == "__main__":
    # main()
    url = r"https://www.google.com/maps/place/?q=place_id:ChIJ7WjSWynClEARUUiva4PiDzI"
    url2 = r"https://www.google.ru/maps/place/LOTTE+HOTEL+ST.+PETERSBURG/@59.9313986,30.2898216,14z/data=!4m11!3m10!1s0x469631034b662bf1:0x71def80ee9724829!5m2!4m1!1i2!8m2!3d59.931402!4d30.310422!9m1!1b1!16s%2Fg%2F11c6d_l0s2?entry=ttu&g_ep=EgoyMDI1MDQwMi4xIKXMDSoJLDEwMjExNjM5SAFQAw%3D%3D"
    # google_maps_parse(url, sorting='new', collect_extra=True,
    #                   min_date=datetime(year=2024, month=1, day=1))
