import time
from datetime import datetime

import pandas as pd
import undetected_chromedriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from .config import OtzovikConfig
from ...abstract import Parser
from ...core import MasterParserConfig


class OtzovikParser(Parser):
    def __init__(self, local_config: OtzovikConfig):
        super().__init__(2, local_config)

    def parse(
        self, global_config: MasterParserConfig
    ) -> list[dict[str, str | int | float | None]]:
        min_date = self._date_convert(global_config.min_date, datetime)
        max_date = self._date_convert(global_config.max_date, datetime)
        count_items = global_config.count_items
        
        driver = self.__initialize_browser(self.config.q)
        review_links, review_dates = self.__collect_review_links(
            driver, min_date, max_date)
        
        official = driver.find_elements(
            By.CSS_SELECTOR, "div.otz_product_header_left > a.product-official"
        )
        official = official[-1].get_attribute("href") if official else None

        data = []
        for link, date in zip(review_links, review_dates):
            review = self.__get_review_data(driver, link, date, official)
            if review is None:
                continue

            data.append(review)
            if count_items != -1 and len(data) >= count_items:
                break
        
        driver.close()
        return data

    def __initialize_browser(self, url):
        driver = undetected_chromedriver.Chrome()
        driver.get(url if url.startswith("http") else "https://otzovik.com/" + url)
        time.sleep(2)
        while self.__check_captcha(driver):
            print("Please, enter captcha to continue parsing Otzovik")
            time.sleep(5)

        return driver

    @staticmethod
    def __check_captcha(driver):
        return driver.find_elements(By.CSS_SELECTOR, "input[name='captcha_url']")

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

    def __collect_review_links(self, driver, min_date, max_date):
        self.__click_element(driver, value="span.tab.neg > a")
        time.sleep(2)
        self.__click_element(driver, value="#reviews-sort-tools-btn")
        time.sleep(0.5)

        # driver.find_element(By.XPATH,
        #                     "//input[@type='radio' and @value='date_desc']")
        sort_buttons = driver.find_elements(By.CSS_SELECTOR, "label")
        for btn in sort_buttons:
            if "Сначала новые" in btn.text:
                btn.find_element(By.CSS_SELECTOR, "span.radio").click()

        self.__click_element(driver, value="button", find_value="Применить")
        time.sleep(3)

        reviews = driver.find_elements(By.CSS_SELECTOR, "div[itemprop='review']")
        links = []
        dates = []
        for review in reviews:
            date = review.find_element(
                By.CSS_SELECTOR,
                "div.item-right > div.rating-wrap > div.review-postdate"
            )
            date = (datetime.strptime(date.get_attribute("content"),
                                     "%Y-%m-%dT%H:%M:%S%z")
                    .replace(tzinfo=None))
            if ((min_date is not None and date < min_date)
               or (max_date is not None and date > max_date)):
                continue
            
            links.append(review.find_element(
                By.CSS_SELECTOR, "a.review-btn.review-read-link"
            ).get_attribute("href"))
            dates.append(date)
            
        next_page = driver.find_elements(By.CSS_SELECTOR, "div.pager > div > a.next")
        if not driver.find_elements(By.CSS_SELECTOR, "div.pager") or not next_page:
            return links, dates

        while next_page:
            reviews = driver.find_elements(
                By.CSS_SELECTOR, "div[itemprop='review']"
            )
            for review in reviews:
                date = review.find_element(
                    By.CSS_SELECTOR,
                    "div.item-right > div.rating-wrap > div.review-postdate"
                )
                date = datetime.strptime(date.get_attribute("content"),
                                         "%Y-%m-%dT%H:%M:%S%z")
                if ((min_date is not None and date < min_date)
                    or (max_date is not None and date > max_date)):
                    continue
                
                links.append(review.find_element(
                    By.CSS_SELECTOR, "a.review-btn.review-read-link"
                ).get_attribute("href"))
                dates.append(date)

            self.__click_element(driver, value="div.pager > div > a.next")
            time.sleep(2)
            next_page = driver.find_elements(
                By.CSS_SELECTOR, "div.pager > div > a.next"
            )

        return links, dates

    def __get_review_data(self, driver, url, date, official=None):
        driver.get(url)
        time.sleep(1)
        while self.__check_captcha(driver):
            print("Please, enter captcha to continue parsing Otzovik")
            time.sleep(5)

        # date = driver.find_element(
        #     By.CSS_SELECTOR, "span.review-postdate.dtreviewed > abbr"
        # ).get_attribute("title")
        # date = datetime.fromisoformat(date)
        # if (min_date is not None and date < min_date) or (
        #     max_date is not None and date > max_date
        # ):
        #     return None
        #
        # date = date.timestamp()

        user = driver.find_element(
            By.CSS_SELECTOR, "div.login-col > a > span[itemprop='name']"
        ).text
        rating = int(
            driver.find_element(By.CSS_SELECTOR, "abbr.rating").get_attribute("title")
        )
        review_text = driver.find_element(By.CSS_SELECTOR, "div.review-minus").text
        review_text += "\n"
        review_text += driver.find_element(
            By.CSS_SELECTOR, "div.review-body.description"
        ).text

        answer = None
        if official is not None:
            comments = driver.find_elements(By.CSS_SELECTOR, "div#comments")
            if comments:
                comments = comments[-1].find_elements(
                    By.CSS_SELECTOR, "div#comments-container > div > div.comment"
                )
                for comment in comments:
                    profile = comment.find_elements(
                        By.CSS_SELECTOR, "div > div > a.user-login"
                    )
                    if profile:
                        profile = profile[-1].get_attribute("href")
                        if profile == official:
                            answer = comment.find_element(
                                By.CSS_SELECTOR, "div.comment-body"
                            ).text
                            break

        return {
            "service_id": self.service_id,
            "name": user,
            "additional_id": None,
            "date": int(date.timestamp()),
            "rating": rating,
            "text": review_text,
            "answer": answer,
        }


def handle_reviews_data(df):
    return df.sort_values("date", ascending=False)


def save_reviews_to_csv(reviews, filename="otzovik_reviews.csv"):
    if not reviews:
        print("There is no data collected from otzovik.")
        return

    df = pd.DataFrame(reviews)
    try:
        df = handle_reviews_data(df)
    except Exception as e:
        raise e

    df.to_csv(filename, index=False, encoding="utf-8")


# def otzovik_parse(url, min_date=None, file="otzovik_reviews.csv"):
#     parser = OtzovikParser(OtzovikConfig(url))
#     data = parser.parse(MasterConfig(min_date=min_date))
#     save_reviews_to_csv(data, file)


# if __name__ == "__main__":
#     url = "https://otzovik.com/reviews/sanatoriy_mriya_resort_spa_russia_yalta/"
#     url2 = "https://otzovik.com/reviews/sanatoriy_slavutich_ukraina_alushta/"
#     otzovik_parse(url, datetime(year=2024, month=1, day=1))
#     # otzovik_parse(url2, 'pages_test.csv')
