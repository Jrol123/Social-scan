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


# def initialize_browser(url=None):
#     playwright = sync_playwright().start()
#     browser = playwright.chromium.launch(headless=False)
#     context = browser.new_context()
#     page = context.new_page()
#
#     page.goto("https://www.google.com/maps" if url is None else url)
#     page.wait_for_timeout(4000)
#     return playwright, browser, page


def initialize_browser(url=None):
    # opts = Options() # undetected_chromedriver.ChromeOptions()
    # opts.add_argument('--no-sandbox')
    # opts.add_argument('--disable-dev-shm-usage')
    # opts.add_argument('--headless')
    # opts.add_argument('--disable-gpu')
    # driver = webdriver.Chrome(options=opts, service=Service(
    #     r"C:\Users\Jrytoeku Qtuhtc\.wdm\drivers\chromedriver\win64\134.0.6998.165\chromedriver.exe")) # undetected_chromedriver.Chrome(options=opts)
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
    # print(last_element.find_elements(By.CSS_SELECTOR, "button[data-photo-index]"))
    # if last_element.find_element(By.XPATH, "//span[contains(text(), 'Нравится')]"):
    #     driver.execute_script(
    #         "arguments[0].scrollIntoView()",
    #         reviews_section.find_elements(
    #             By.XPATH, "//span[contains(text(), 'Нравится')]")[-1])
    # elif len(last_element.find_elements(By.CSS_SELECTOR, "button[data-photo-index]")) > 0:
    #     driver.execute_script(
    #         "arguments[0].scrollIntoView()",
    #         last_element.find_elements(
    #             By.CSS_SELECTOR, "button[data-photo-index]")[-1])
    #     print('ЫЫЫ')
    # else:
    #     driver.execute_script("arguments[0].scrollIntoView()", last_element)
    
    # driver.execute_script(
    #     "arguments[0].scrollTop = arguments[0].scrollHeight",
    #     reviews_section)
    time.sleep(2)
    

def expand_reviews(driver):
    try:
        # Пытаемся снова найти кнопки перед кликом
        load_more_buttons = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//button[contains(text(), 'Ещё')]"))
        )
        # Проверяем, есть ли кнопки для нажатия
        if not load_more_buttons:
            print("Нет кнопок 'Ещё' для загрузки.")
            return
        
        # Проходим по каждой кнопке, чтобы кликнуть
        driver.execute_script("arguments[0].scrollIntoView();",
                              load_more_buttons[0])
        time.sleep(1)
        for button in load_more_buttons:
            # Прокручиваем до кнопки
            driver.execute_script("arguments[0].scrollIntoView();", button)
            
            # Проверяем, если кнопка кликабельна
            if button.is_displayed() and button.is_enabled():
                try:
                    button.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", button)
                
                # time.sleep(0.3)
                # break
        else:
            # Если цикл завершился без нахождения кнопки для нажатия
            print("Нет доступных кнопок 'Ещё' для загрузки.")
            return
    
    except Exception as e:
        print("Произошла ошибка:", e)
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
        # review_section = page.get_by_role('tab', name="Отзывы")
        # review_section.click()
        # page.wait_for_timeout(3000)
        #
        # # Choose sorting by date
        # if sorting != 'relevant':
        #     page.locator('text=Самые релевантные').click()
        #     page.wait_for_timeout(1000)
        #
        #     page.locator('text=' + sortings[sorting]).click(force=True)
        #     page.wait_for_timeout(2000)
        
        # Locate and click the reviews section
        # WebDriverWait(driver, 10).until(
        #     EC.presence_of_element_located(
        #         (By.XPATH, "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div/div/button[contains(text(), 'Отзывы')]"))
        # )
        click_element(driver, By.XPATH, "//div[contains(text(), 'Отзывы')]")
        
        # Wait for reviews to load
        # WebDriverWait(driver, 10).until(
        #     EC.presence_of_element_located(
        #         (By.XPATH,
        #          "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div"))
        # )
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
        
        reviews_section = driver.find_element(
            By.XPATH,
            "//*[@id='QA0Szd']/div/div/div[1]/div[2]/div/div[1]/div/div"
        )
        if max_reviews is None:
            # previous_height = driver.execute_script(
            #     "return document.body.scrollHeight")
            
            # while True:
                # last_element = driver.find_elements(
                #     By.CSS_SELECTOR, 'div.AyRUI[aria-hidden="true"]')[-1]
                # driver.execute_script("arguments[0].scrollIntoView();",
                #                       last_element)
                # driver.execute_script(
                #     "arguments[0].scrollTop = arguments[0].scrollHeight",
                #     last_element)
                # for _ in range(5):
                    # driver.execute_script(
                    #     "arguments[0].scrollTop = arguments[0].scrollHeight",
                    #     reviews_section)
                    # driver.execute_script(
                    #     "window.scrollTo(0, document.body.scrollHeight);")
                    # time.sleep(0.3)
                    # print('sss')
                    
                # time.sleep(2.5)  # Задержка для загрузки отзывов
                #
                # new_height = driver.execute_script(
                #     "return document.body.scrollHeight")
                # if new_height == previous_height:
                #     break
                #
                # previous_height = new_height
            
            prev_element = driver.find_elements(By.CSS_SELECTOR,
                                                "div[class*='jJc9Ad']")[-1]
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
        # for _ in range(max_reviews//10 + 1):
        #     page.mouse.wheel(0, 5000)
        #     # page.wait_for_timeout(2000)
        #
        #     try:
        #         expand_reviews = page.locator('button:has-text("Ещё")')
        #         expand_offset = 0
        #         i = 0
        #         while expand_reviews.count() > expand_offset:
        #             element = expand_reviews.all()[expand_offset]
        #             try:
        #                 element.click(timeout=3000)
        #             except Exception:
        #                 # print('error')
        #                 expand_offset += 1
        #
        #             page.wait_for_timeout(500)
        #             # page.mouse.wheel(0, 1000)
        #             expand_reviews = page.locator('button:has-text("Ещё")')
        #             i += 1
        #             if i - expand_offset > max_reviews:
        #                 break
        #
        #     except Exception:
        #         pass
        #
        #     page.mouse.wheel(0, 1000)
        #     page.wait_for_timeout(2000)


        # Extract reviews
        # review_elements = page.locator("div[class*='jJc9Ad']")
        review_elements = driver.find_elements(By.CSS_SELECTOR,
                                               "div[class*='jJc9Ad']")
        # logger.info(f"Found {review_elements.count()} reviews")
        for element in review_elements:
            # reviewer = element.locator("div[class*='d4r55']").inner_text()
            # rating = element.locator("span[class*='fzvQIb']").inner_text()
            # date = element.locator("span[class*='xRkPPb']").inner_text().rsplit(',', 1)[0]
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

def google_maps_parse(object, max_reviews=100, file="google_reviews.csv"):
    driver = initialize_browser(object)
    try:
        # search_google_maps(page, object)
        reviews = scrape_reviews(driver, max_reviews=max_reviews, sorting='new', collect_extra=True)
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
