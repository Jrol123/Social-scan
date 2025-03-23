import emoji
import logging
import pandas as pd
import re
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


def initialize_browser(url=None):
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("https://www.google.com/maps" if url is None else url)
    page.wait_for_timeout(4000)
    return playwright, browser, page


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

def scrape_reviews(page, max_reviews=100, sorting='relevant', collect_extra=False):
    assert sorting in ('relevant', 'new', 'increase', 'decrease')

    sortings = {'relevant': 'Самые релевантные', 'new': 'Сначала новые', 
                'increase': 'По возрастанию рейтинга', 'decrease': 'По убыванию рейтинга'}
    reviews = []
    try:
        # Wait for the business details to load
        page.wait_for_timeout(4000)
        
        # Locate and click the reviews section
        # logger.info("Searching for reviews section")
        review_section = page.get_by_role('tab', name="Отзывы")
        review_section.click()
        page.wait_for_timeout(3000)

        # Choose sorting by date
        if sorting != 'relevant':
            page.locator('text=Самые релевантные').click()
            page.wait_for_timeout(1000)

            page.locator('text=' + sortings[sorting]).click(force=True)
            page.wait_for_timeout(2000)

        # Scroll to load more reviews
        # logger.info("Loading reviews...")
        for _ in range(max_reviews//10 + 1):
            page.mouse.wheel(0, 5000)
            # page.wait_for_timeout(2000)

            try:
                expand_reviews = page.locator('button:has-text("Ещё")')
                expand_offset = 0
                i = 0
                while expand_reviews.count() > expand_offset:
                    element = expand_reviews.all()[expand_offset]
                    try:
                        element.click(timeout=3000)
                    except Exception:
                        # print('error')
                        expand_offset += 1
                    
                    page.wait_for_timeout(500)
                    # page.mouse.wheel(0, 1000)
                    expand_reviews = page.locator('button:has-text("Ещё")')
                    i += 1
                    if i - expand_offset > max_reviews:
                        break
                
            except Exception:
                pass
            
            page.mouse.wheel(0, 1000)
            page.wait_for_timeout(2000)

        # for _ in range(max_reviews//10):
        #     page.mouse.wheel(0, 5000)
        #     page.wait_for_timeout(1500)

        # expand_reviews = page.locator('button:has-text("Ещё")')
        # expand_offset = 0
        # # i = 0
        # while expand_reviews.count() > expand_offset:
        #     element = expand_reviews.all()[expand_offset]
        #     try:
        #         element.click(timeout=3000)
        #     except Exception:
        #         # print('error')
        #         expand_offset += 1
            
        #     page.wait_for_timeout(500)
        #     # page.mouse.wheel(0, 1000)
        #     expand_reviews = page.locator('button:has-text("Ещё")')
        #     # print(element)
        #     # print('ммм' + str(i))
        #     # i += 1

        # Extract reviews
        review_elements = page.locator("div[class*='jJc9Ad']")
        # logger.info(f"Found {review_elements.count()} reviews")
        for element in review_elements.all():
            reviewer = element.locator("div[class*='d4r55']").inner_text()
            rating = element.locator("span[class*='fzvQIb']").inner_text()
            date = element.locator("span[class*='xRkPPb']").inner_text().rsplit(',', 1)[0]
            
            try:
                if collect_extra:
                    review_text = element.locator("div[class*='MyEned']").inner_text(timeout=2000)
                else:
                    review_text = element.locator("span[class*='wiI7pd']").inner_text(timeout=2000)
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
    playwright, browser, page = initialize_browser(object)
    try:
        # search_google_maps(page, object)
        reviews = scrape_reviews(page, max_reviews=max_reviews, collect_extra=True)
        save_reviews_to_csv(reviews, file)
    finally:
        page.wait_for_timeout(5000)
        browser.close()
        playwright.stop()

def main():
    # business_name = "МРИЯ РЕЗОРТ энд СПА"
    # business_name = "Дальневосточный федеральный университет, Русский, Приморский край"
    business_url = r"https://www.google.ru/maps/place/МРИЯ+РЕЗОРТ+энд+СПА/@44.393895,33.9240737,14z/data=!4m22!1m12!3m11!1s0x4094c2295bd268ed:0x320fe2836baf4851!2z0JzQoNCY0K8g0KDQldCX0J7QoNCiINGN0L3QtCDQodCf0JA!5m2!4m1!1i2!8m2!3d44.3969251!4d33.9396257!9m1!1b1!16s%2Fg%2F1q65ck494!3m8!1s0x4094c2295bd268ed:0x320fe2836baf4851!5m2!4m1!1i2!8m2!3d44.3969251!4d33.9396257!16s%2Fg%2F1q65ck494?entry=ttu&g_ep=EgoyMDI1MDMxOS4yIKXMDSoJLDEwMjExNjM5SAFQAw%3D%3D"

    # Initialize browser
    playwright, browser, page = initialize_browser(business_url)
    
    try:
        # Search and scrape reviews
        # search_google_maps(page, business_name)
        reviews = scrape_reviews(page, max_reviews=50, collect_extra=True)
        
        # Save results
        save_reviews_to_csv(reviews)
    
    except Exception as e:
        # logger.error(f"Unexpected error: {e}")
        raise e
    
    finally:
        # Add a longer wait before closing
        page.wait_for_timeout(5000)
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    main()
