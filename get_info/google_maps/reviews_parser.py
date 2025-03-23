from playwright.sync_api import sync_playwright
import pandas as pd
import re
import emoji
import logging
# import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_browser():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    return playwright, browser, page

def search_google_maps(page, business_name):
    page.goto("https://www.google.com/maps")
    search_box = page.locator("input[id='searchboxinput']")
    search_box.fill(business_name)
    search_box.press("Enter")
    page.wait_for_timeout(4000)

def clean_text(text):
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove extra whitespace
    text = re.sub(r's+', ' ', text).strip()
    
    return text

def scrape_reviews(page, max_reviews=100):
    reviews = []
    try:
        # Wait for the business details to load
        page.wait_for_timeout(5000)
        
        # Locate and click the reviews section
        logger.info("Searching for reviews section")
        review_section = page.get_by_role('tab', name="Отзывы")
        review_section.click()
        page.wait_for_timeout(3000)

        # Choose sorting by date
        page.locator('text=Самые релевантные').click()
        page.wait_for_timeout(1000)

        page.locator('text=Сначала новые').click(force=True)
        page.wait_for_timeout(2000)

        # Scroll to load more reviews
        logger.info("Loading reviews...")
        for _ in range(max_reviews//10 + 1):
            page.mouse.wheel(0, 5000)
            # page.wait_for_timeout(2000)

            try:
                for element in page.locator('text=Ещё').all():
                    # print(element)
                    element.click(timeout=1000)
                    page.wait_for_timeout(250)
            except Exception:
                pass

            page.wait_for_timeout(2000)

        # Extract reviews
        review_elements = page.locator("div[class*='jJc9Ad']")
        logger.info(f"Found {review_elements.count()} reviews")
        for element in review_elements.all():
            reviewer = element.locator("div[class*='d4r55']").inner_text()
            rating = element.locator("span[class*='fzvQIb']").inner_text()
            date = element.locator("span[class*='xRkPPb']").inner_text().rsplit(',', 1)[0]
            try:
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
        logger.error(f"Error during scraping: {e}")
    
    return reviews

def save_reviews_to_csv(reviews, filename="google_reviews.csv"):
    df = pd.DataFrame(reviews)
    df.to_csv(filename, index=False, encoding='utf-8')
    logger.info(f"Reviews saved to {filename}")

def main():
    business_name = "МРИЯ РЕЗОРТ энд СПА"
    
    # Initialize browser
    playwright, browser, page = initialize_browser()
    
    try:
        # Search and scrape reviews
        search_google_maps(page, business_name)
        reviews = scrape_reviews(page, max_reviews=50)
        
        # Save results
        save_reviews_to_csv(reviews)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        # Add a longer wait before closing
        page.wait_for_timeout(5000)
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    main()