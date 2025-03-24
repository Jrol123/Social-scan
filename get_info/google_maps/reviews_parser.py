import emoji
# import logging
import pandas as pd
import re
import time
# import undetected_chromedriver
from datetime import datetime, timedelta

from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


def initialize_browser(url=None):
    driver = webdriver.Chrome()
    driver.get("https://www.google.com/maps" if url is None else url)
    # time.sleep(2)
    return driver

# def search_google_maps(page, business_name):
#     page.goto("https://www.google.com/maps")
#     search_box = page.locator("input[id='searchboxinput']")
#     search_box.fill(business_name)
#     search_box.press("Enter")
#     page.wait_for_timeout(4000)
    
    # search_results = page.locator("a[class*='hfpxzc']")
    # print(search_results.count())
    # if search_results.count() > 0:
    #     res = search_results.first
    #     res.click()
    #     page.wait_for_timeout(3000)

def clean_text(text):
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove extra whitespace
    text = re.sub(r's+', ' ', text).strip()
    
    return text

def click_element(driver, by=By.CSS_SELECTOR, value=None, find_value=None):
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

def scroll_reviews(driver, reviews_section):
    last_element = driver.find_elements(
        By.CSS_SELECTOR, 'div.AyRUI[aria-hidden="true"]')[-1]
    driver.execute_script("arguments[0].scrollIntoView();",
                          last_element)
    time.sleep(2)
    
def expand_reviews(driver):
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
            print("There are no any 'Ещё' buttons to load.")
            return
    
    except Exception as e:
        print("Error occurred:", e)
        return

def scrape_reviews(driver, max_reviews=None, sorting='relevant', collect_extra=False):
    assert sorting in ('relevant', 'new', 'increase', 'decrease')

    sortings = {'relevant': 'Самые релевантные', 'new': 'Сначала новые', 
                'increase': 'По возрастанию рейтинга', 'decrease': 'По убыванию рейтинга'}
    reviews = []
    try:
        # Wait for the business details to load
        # time.sleep(1)
        
        # # Locate and click the reviews section
        # # logger.info("Searching for reviews section")
        click_element(driver, By.XPATH, "//div[contains(text(), 'Отзывы')]")
        time.sleep(1)
        
        # Choose sorting type
        if sorting != 'relevant':
            click_element(driver, By.XPATH, "//div[contains(text(), 'Самые релевантные')]")
            time.sleep(0.5)
            click_element(driver, By.XPATH,
                          f"//div[contains(text(), '{sortings[sorting]}')]")
            time.sleep(3)

        # Scroll to load more reviews
        # logger.info("Loading reviews...")
        reviews_section = driver.find_element(By.XPATH,
            "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div")
        if max_reviews is None:
            prev_element = driver.find_elements(By.CSS_SELECTOR,
                                                "div[class*='jJc9Ad']")[-1]
            
            # # Получение общего числа отзывов.
            # # Возможно понадобится для проверки на пасинг всех отзывов со страницы
            # total_reviews = driver.find_element(
            #     By.XPATH, "//div[contains(text(), 'Отзывов:')]").text
            # total_reviews = int(total_reviews.split(': ', 1)[1].replace(' ', '')
            #                     .replace('&nbsp;', '').strip()) // 10
            # print(total_reviews * 10)
            i = 0
            while True:
                scroll_reviews(driver, reviews_section)
                # time.sleep(3)
                curr_element = driver.find_elements(By.CSS_SELECTOR,
                                                    "div[class*='jJc9Ad']")[-1]
                time.sleep(2)
                if prev_element == curr_element:
                    # Waiting for next reviews to load
                    for _ in range(120):
                        time.sleep(1)
                        curr_element = driver.find_elements(
                            By.CSS_SELECTOR, "div[class*='jJc9Ad']")[-1]
                        if prev_element != curr_element:
                            break
                    else: # If time is expired, consider we reached the bottom
                        print(i)
                        print(curr_element.text)
                        break
                else:
                    i -= 1

                prev_element = curr_element
                i += 1
        else:
            for _ in range(max_reviews // 10 - 1):
                scroll_reviews(driver, reviews_section)
        
        expand_reviews(driver)
        
        # Extract reviews
        review_elements = driver.find_elements(By.CSS_SELECTOR,
                                               "div[class*='jJc9Ad']")
        # logger.info(f"Found {review_elements.count()} reviews")
        for element in review_elements:
            reviewer = element.find_element(By.CSS_SELECTOR,
                                            "div[class*='d4r55']").text
            rating = element.find_element(By.CSS_SELECTOR,
                                          "span[class*='fzvQIb']").text
            date = element.find_element(By.CSS_SELECTOR,
                                             "span[class*='xRkPPb']").text
            date = date.rsplit(',', 1)[0].strip()
            
            if collect_extra:
                text_selector = "div[class*='MyEned']"
            else:
                text_selector = "span[class*='wiI7pd']"
                
            try:
                review_text = element.find_element(
                    By.CSS_SELECTOR, text_selector).text
            except Exception:
                continue

            reviews.append({
                "user": clean_text(reviewer),
                "rating": rating,
                "date": date,
                "review": clean_text(review_text)
            })
       
    except Exception as e:
        raise e
        # logger.error(f"Error during scraping: {e}")
    
    return reviews

def handle_reviews_data(df):
    time_units = {'день': timedelta(days=1), 'дн': timedelta(days=1), 'недел': timedelta(weeks=1)}
    now = datetime.now()

    def text_to_date(text):
        nonlocal time_units, now

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

            date = now.replace(year = now.year if now.month > num else now.year - 1, 
                            month = now.month - num if now.month > num else  now.month - num + 12)
        else:
            for key in time_units:
                if unit.startswith(key):
                    unit = key
                    break

            date = now - num*time_units[unit]

        return date.timestamp()

    df['rating'] = df['rating'].apply(lambda x: int(x.split(' ')[0]))
    df['date'] = df['date'].str.lower().apply(text_to_date)
    df['review'] = df['review'].str.replace('\n', ' ').str.replace('\t', ' ')
    return df

def save_reviews_to_csv(reviews, filename="google_reviews.csv"):
    df = pd.DataFrame(reviews)
    try:
        df = handle_reviews_data(df)
    except Exception as e:
        raise e

    df.to_csv(filename, index=False, encoding='utf-8')
    # logger.info(f"Reviews saved to {filename}")

def google_maps_parse(object, max_reviews=100, sorting='new', collect_extra=True,
                      file="google_reviews.csv"):
    driver = initialize_browser(object)
    try:
        # search_google_maps(page, object)
        reviews = scrape_reviews(driver, max_reviews=max_reviews,
                                 sorting=sorting, collect_extra=collect_extra)
        save_reviews_to_csv(reviews, file)
    finally:
        time.sleep(5)
        driver.quit()

def main():
    # business_name = "МРИЯ РЕЗОРТ энд СПА"
    # business_name = "Дальневосточный федеральный университет, Русский, Приморский край"
    # business_url = r"https://www.google.ru/maps/place/МРИЯ+РЕЗОРТ+энд+СПА/@44.393895,33.9240737,14z/data=!4m22!1m12!3m11!1s0x4094c2295bd268ed:0x320fe2836baf4851!2z0JzQoNCY0K8g0KDQldCX0J7QoNCiINGN0L3QtCDQodCf0JA!5m2!4m1!1i2!8m2!3d44.3969251!4d33.9396257!9m1!1b1!16s%2Fg%2F1q65ck494!3m8!1s0x4094c2295bd268ed:0x320fe2836baf4851!5m2!4m1!1i2!8m2!3d44.3969251!4d33.9396257!16s%2Fg%2F1q65ck494?entry=ttu&g_ep=EgoyMDI1MDMxOS4yIKXMDSoJLDEwMjExNjM5SAFQAw%3D%3D"
    business_url = r"https://www.google.com/maps/place/?q=place_id:ChIJ7WjSWynClEARUUiva4PiDzI"

    # Initialize browser
    driver = initialize_browser(business_url)
    
    try:
        # Search and scrape reviews
        # search_google_maps(page, business_name)
        reviews = scrape_reviews(driver, max_reviews=None, sorting='new', collect_extra=False)
        
        # Save results
        save_reviews_to_csv(reviews)
    
    except Exception as e:
        # logger.error(f"Unexpected error: {e}")
        raise e
    
    finally:
        time.sleep(5)
        driver.quit()


if __name__ == "__main__":
    main()
