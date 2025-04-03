import pandas as pd
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


def initialize_browser(url=None):
    driver = webdriver.Chrome()
    driver.get("https://otzovik.com/" if url is None else url)
    time.sleep(2)
    while check_captcha(driver):
        print('Please, enter captcha to continue parsing Otzovik')
        time.sleep(5)
    
    return driver

def check_captcha(driver):
    return driver.find_elements(By.CSS_SELECTOR, "input[name='captcha_url']")

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

def collect_review_links(driver):
    click_element(driver, value="span.tab.neg > a")
    time.sleep(2)
    click_element(driver, value="#reviews-sort-tools-btn")
    time.sleep(0.5)
    
    # driver.find_element(By.XPATH,
    #                     "//input[@type='radio' and @value='date_desc']")
    sort_buttons = driver.find_elements(By.CSS_SELECTOR, "label")
    for btn in sort_buttons:
        if 'Сначала новые' in btn.text:
            btn.find_element(By.CSS_SELECTOR, "span.radio").click()
    
    click_element(driver, value="button", find_value='Применить')
    time.sleep(3)
    
    reviews = driver.find_elements(By.CSS_SELECTOR,
                                   "a.review-btn.review-read-link")
    links = [link.get_attribute('href') for link in reviews]
    
    next_page = driver.find_elements(By.CSS_SELECTOR, "div.pager > div > a.next")
    if not driver.find_elements(By.CSS_SELECTOR, "div.pager") or not next_page:
        return links
    
    while next_page:
        reviews = driver.find_elements(By.CSS_SELECTOR,
                                       "a.review-btn.review-read-link")
        links.extend([link.get_attribute('href') for link in reviews])
        
        click_element(driver, value="div.pager > div > a.next")
        time.sleep(2)
        next_page = driver.find_elements(By.CSS_SELECTOR,
                                         "div.pager > div > a.next")
        
    return links


def get_review_data(driver, url):
    driver.get(url)
    time.sleep(1)
    
    user = driver.find_element(By.CSS_SELECTOR,
                               "div.login-col > a > span[itemprop='name']").text
    date = (driver.find_element(By.CSS_SELECTOR,
                                "span.review-postdate.dtreviewed > abbr")
            .get_attribute('title'))
    rating = (driver.find_element(By.CSS_SELECTOR, "abbr.rating")
              .get_attribute('title'))
    review_text = driver.find_element(By.CSS_SELECTOR, "div.review-minus").text
    review_text += "\n"
    review_text += driver.find_element(By.CSS_SELECTOR,
                                       "div.review-body.description").text
    return {'user': user, 'rating': rating, 'date': date, 'review': review_text}


def scrap_reviews(url):
    driver = initialize_browser(url)
    review_links = list(set(collect_review_links(driver)))
    data = []
    for link in review_links:
        data.append(get_review_data(driver, link))

    return data

def handle_reviews_data(df):
    df['rating'] = df['rating'].astype(int)
    
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    df = df.sort_values('date', ascending=False)
    return df

def save_reviews_to_csv(reviews, filename="otzovik_reviews.csv"):
    df = pd.DataFrame(reviews)
    try:
        df = handle_reviews_data(df)
    except Exception as e:
        raise e

    df.to_csv(filename, index=False, encoding='utf-8')

def otzovik_parse(url, file="otzovik_reviews.csv"):
    data = scrap_reviews(url)
    save_reviews_to_csv(data, file)

if __name__ == '__main__':
    url = 'https://otzovik.com/reviews/sanatoriy_mriya_ukraina_evpatoriya/'
    url2 = 'https://otzovik.com/reviews/sanatoriy_slavutich_ukraina_alushta/'
    otzovik_parse(url)
