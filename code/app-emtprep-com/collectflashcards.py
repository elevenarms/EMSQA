from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import time
import re
import os
import json

# ❌ Close "Question of the Day" popup
def close_qotd_popup(driver):
    try:
        cancel_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'btn-neutral-outline') and text()='Cancel']"))
        )
        driver.execute_script("arguments[0].click();", cancel_button)
        print("Popup dismissed by clicking Cancel.")
    except TimeoutException:
        print("No 'Question of the Day' popup appeared.")


# 🔄 Function to safely click an element
def safe_click(driver, element, retries=3):
    for attempt in range(retries):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except StaleElementReferenceException:
            print(f"Stale element click failed, retrying {attempt + 1}/{retries}...")
            time.sleep(2)
    return False


# 🧐 Function to fetch a fresh element to avoid StaleElementReferenceException
def get_fresh_element(driver, xpath, retries=3):
    for attempt in range(retries):
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            return element
        except (StaleElementReferenceException, TimeoutException):
            print(f"Element fetch failed on attempt {attempt + 1}/{retries}. Retrying...")
            time.sleep(2)
    raise Exception(f"Failed to fetch element after {retries} retries.")

def get_flashcard_topics(driver):
    """Fetches all flashcard topics after each navigation."""
    return driver.find_elements(By.XPATH, "//div[contains(@class, 'hover:cursor-pointer') and contains(@class, 'flex gap-2')]")


def get_next_page_button(driver):
    """Finds the specific 'Next Page' button and ensures it's clickable."""
    try:
        # Find all pagination buttons
        buttons = driver.find_elements(By.XPATH, "//div[contains(@class, 'rounded bg-primary-default') and contains(@class, 'cursor-pointer')]")
        
        if len(buttons) > 1:
            next_button = buttons[-2]  # Select second-to-last button (should be "Next")
        elif buttons:
            next_button = buttons[0]  # If only one button exists, use it
        else:
            return None  # No next page button found

        # Check if the button is disabled
        if "btn-disabled" in next_button.get_attribute("class"):
            print("🛑 'Next Page' button is disabled. Reached last page.")
            return None  # Stop pagination

        return next_button

    except Exception as e:
        print(f"❌ Error finding next page button: {e}")
        return None



# 📌 Function to loop over flashcards and extract content
def flashcard_crawl(level):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Run without UI (Enable for headless mode)
    options.add_argument("--disable-gpu")  # Fix rendering issues
    options.add_argument("--window-size=1920,1080")  # Set resolution
    driver = webdriver.Chrome(options=options)

    # Open the Flashcard Page
    driver.get("https://app.emtprep.com/student/flashcards")
    wait = WebDriverWait(driver, 10)

    # ✅ Login process
    username_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
    password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))

    # 🔑 Enter credentials
    username_field.send_keys("zar8jw@virginia.edu")  # Replace with actual email
    password_field.send_keys("X39SnHjp@w@J49g")  # Replace with actual password
    password_field.send_keys(Keys.RETURN)

    time.sleep(5)
    close_qotd_popup(driver)

    # 🚀 Ensure login success
    if "login" in driver.current_url:
        print("Login failed. Please check your credentials.")
        driver.quit()
        exit()

    time.sleep(3)

    while True:
        # ⏳ Wait for flashcards to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'hover:cursor-pointer') and contains(@class, 'flex gap-2')]"))
        )

        # 🔍 Get initial list of flashcard topics
        flashcard_topics = get_flashcard_topics(driver)

        if not flashcard_topics:
            print("❌ No flashcard topics found. Check XPath or ensure flashcards are visible.")
            driver.quit()
            return

        print(f"✅ Found {len(flashcard_topics)} flashcard topics.")

        flashcards = {}
        # 🔄 Loop through each flashcard topic
        for i in range(len(flashcard_topics)):
            # 🔄 Re-fetch topics after navigation
            flashcard_topics = get_flashcard_topics(driver)

            if i >= len(flashcard_topics):
                print("⚠️ Topic index out of range after re-fetching. Skipping...")
                continue

            topic = flashcard_topics[i]  # Ensure correct indexing
            # Extract the topic name from the current flashcard topic element
            topic_name = flashcard_topics[i].find_element(
                By.XPATH, ".//div[contains(@class, 'text-[16px]')]/div"
            ).text.strip()
            topic_name = topic_name.replace("/", " or ")  # Replace spaces with underscores
            if not os.path.exists(f"../../log/flashcards/{level}"):
                os.makedirs(f"../../log/flashcards/{level}")
            if f"{topic_name}.json" in os.listdir(f"../../log/flashcards/{level}/"):
                print(f"❌ Flashcards for '{topic_name}' already exist. Skipping...")
                with open(f"../../log/flashcards/{level}/{topic_name}.json", "r") as f:
                    flashcards[topic_name] = json.load(f)
                continue

            print(f"🔹 Clicking flashcard topic {i+1}/{len(flashcard_topics)}...")
            
            # Extract the number of flashcards
            num_cards_text = topic.find_element(By.XPATH, ".//div[@class='text-[14px] font-medium']/span").text
            num_cards = int(re.search(r'\d+', num_cards_text).group())  # Extract only the number

            print(f"   ➡️ Expected {num_cards} flashcards in this topic.")
            # Scroll into view & Click flashcard topic
            driver.execute_script("arguments[0].scrollIntoView();", topic)
            safe_click(driver, topic)
            
            time.sleep(3)  # Wait for flashcards to load

            # Validate flashcards are loading by checking for the first card
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'relative mx-auto mb-8')]"))
            )

            flashcard = {}
            # 🔄 Loop through each flashcard
            for j in range(num_cards):
                # Extract front (question) side of the flashcard
                front_text_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'whitespace-pre-wrap')]"))
                )
                front_text = front_text_element.text.strip()
                print(f"   📖 Front (Question): {front_text}")
                
                body_element = driver.find_element(By.TAG_NAME, "body")
                body_element.send_keys(Keys.ARROW_RIGHT)  # Press right arrow key
                time.sleep(2)  # Wait for flip animation

                # Extract back (answer) side of the flashcard
                back_text_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'whitespace-pre-wrap')]"))
                )
                back_text = back_text_element.text.strip()
                print(f"   📝 Back (Answer): {back_text}")
                
                # Record the flashcard in the dictionary
                flashcard[front_text] = back_text

                # Locate the green button (proceed to the next flashcard)
                green_button = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'text-score-great-default')]"))
                )
                safe_click(driver, green_button)
                time.sleep(3)  # Wait for new flashcard to load

            # Locate "Back to Dashboard" button with a better XPath

            # ✅ Wait for the score element to appear
            score_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Score')]/following-sibling::div"))
            )
            # 🎯 Extract score text
            score_text = score_element.text.strip()  # Example: "100%"
            assert score_text == "100%", f"Score is not 100%: {score_text}"

            with open(f"../../log/flashcards/{level}/{topic_name}.json", "w") as f:
                json.dump(flashcard, f, indent=4)

            flashcards[topic_name] = flashcard

            dashboard_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'btn-primary') and contains(text(), 'Dashboard')]"))
            )
            driver.execute_script("arguments[0].click();", dashboard_button)  # Ensure the button clicks even if hidden
            time.sleep(3)  # Wait for navigation
            print("✅ Clicked 'Back to Dashboard' successfully!")
        
        # 🔄 **Check if there is a "Next Page" button**
        next_page_button = get_next_page_button(driver)
        if next_page_button:
            print("⏭️ Clicking 'Next Page' to load more flashcard topics...")
            safe_click(driver, next_page_button)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'hover:cursor-pointer') and contains(@class, 'flex gap-2')]"))
            )
            time.sleep(5)  # Wait for new topics to load
        else:
            print("✅ No more pages left. Flashcard extraction complete!")
            break  # **Exit while-loop** if no more pages


    # ✅ Close the browser
    driver.quit()
    print("✅ Flashcard extraction complete!")
    if not os.path.exists("../../knowledge/app-emtprep-com/flashcards"):
        os.makedirs("../../knowledge/app-emtprep-com/flashcards")
    with open(f"../../knowledge/app-emtprep-com/flashcards/{level}.json", "w") as f:
        json.dump(flashcards, f, indent=4)


# 🚀 Run the flashcard crawler
if __name__ == "__main__":
    level = "Critical Care"
    flashcard_crawl(level)
