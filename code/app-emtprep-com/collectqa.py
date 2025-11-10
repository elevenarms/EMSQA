from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import json
import re
import requests
import os
from html2qa import parse_html, rex_extract


def memory_load(level, section):
    if not os.path.exists(f"../../log/quiz/app-emtprep-com/{level}/{section}/memory.json"):
        return {}
    else:
        with open(f"../../log/quiz/app-emtprep-com/{level}/{section}/memory.json", "r") as f:
            mem = json.load(f)
        return mem


def update_memory(trial, level, section):
    memory = memory_load(level, section)

    new_q_cnt = 0
    for file in os.listdir(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}"):
        if file.endswith(".json"):
            with open(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}/{file}", "r") as f:
                data = json.load(f)


            q = data["question"].strip().lower()

            if q not in memory:
                new_q_cnt += 1
                memory[q] = {
                    "choices": data["choices"],
                    "answer": data["answer"],
                    "explanation": data["explanation"],
                }

    with open(f"../../log/quiz/app-emtprep-com/{level}/{section}/memory.json", "w") as f:
        json.dump(memory, f, indent=4)
    print("*" * 50)
    print(f"Memory updated. Trial {trial}: #New/#Total: {new_q_cnt}/{len(memory)}")
    print("*" * 50)

# 🔄 Retry fetching element
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


# 🔄 Safe click with retries
def safe_click(driver, element, retries=3):
    for attempt in range(retries):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except StaleElementReferenceException:
            print(f"Stale element click failed, retrying {attempt + 1}/{retries}...")
            time.sleep(2)
    return False


# ❌ Close "Question of the Day" popup
def close_qotd_popup(driver):
    try:
        cancel_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((
                By.XPATH, "//div[contains(@class, 'btn-neutral-outline') and text()='Cancel']"
            ))
        )
        driver.execute_script("arguments[0].click();", cancel_button)
        print("Popup dismissed by clicking Cancel.")
    except TimeoutException:
        print("No 'Question of the Day' popup appeared.")


# ⏳ Wait for a new question to load without refreshing prematurely
def wait_for_new_question(driver, old_text, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            current_question_element = driver.find_element(
                By.XPATH, "//div[contains(@class, 'mb-8') and contains(@class, 'text-gray-900')]/span"
            )
            current_text = current_question_element.text.strip().lower()
            if current_text != old_text:
                return current_text  # Successfully loaded a new question
        except StaleElementReferenceException:
            print("Stale element detected. Trying to fetch again...")
        attempt += 1
        time.sleep(1)

    return None  # Failed after retries


def get_last_trial_number(directory):
    pattern = re.compile(r"trial-(\d+)")
    max_num = -1

    for folder in os.listdir(directory):
        match = pattern.match(folder)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num if max_num != -1 else None  # Return None if no matching folder is found


# 🚀 Main web-crawling function
def web_crawl(trial, level, section):
    mem = memory_load(level, section)
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Run without UI
    options.add_argument("--disable-gpu")  # Fix rendering issues
    options.add_argument("--window-size=1920,1080")  # Set resolution
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Network.enable", {})

    driver.get("https://app.emtprep.com/student/dashboard")
    wait = WebDriverWait(driver, 10)

    # Login process
    username_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
    password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))

    # Enter credentials
    username_field.send_keys("zar8jw@virginia.edu")  # Replace with actual email
    password_field.send_keys("X39SnHjp@w@J49g")  # Replace with actual password
    password_field.send_keys(Keys.RETURN)

    time.sleep(5)
    close_qotd_popup(driver)

    if "login" in driver.current_url:
        print("Login failed. Please check your credentials.")
        driver.quit()
        exit()

    category_element = driver.find_element(By.XPATH, f"//span[@aria-label='{section}']")
    driver.execute_script("arguments[0].scrollIntoView(true);", category_element)

    take_quiz_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((
            By.XPATH, f"//div[contains(@class, 'md:block')]//span[@aria-label='{section}']/ancestor::tr//div[contains(@class, 'btn-primary-outline') and .//span[text()='Take Quiz']]"
        ))
    )

    safe_click(driver, take_quiz_button)
    print(f"Clicked on 'Take Quiz' for {section} (Desktop View).")

    session_cookies = driver.get_cookies()
    session = requests.Session()
    for cookie in session_cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    question_count = 1
    previous_question_text = ""
    retries = 0
    max_retries = 3
    has_new_question = False

    while True:
        try:
            question_text_element = get_fresh_element(driver, "//div[contains(@class, 'mb-8') and contains(@class, 'text-gray-900')]/span")
            current_question_text = question_text_element.text.lower().strip()
            print(f"Current question text: {current_question_text}")

            if current_question_text == previous_question_text:
                retries += 1
                if retries >= max_retries:
                    print("Repeated question detected multiple times. Refreshing page...")
                    driver.refresh()
                    previous_question_text = ""
                    time.sleep(5)
                    retries = 0
                continue

            retries = 0
            previous_question_text = current_question_text

            answer_options = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, "//div[contains(@class, 'hover:cursor-pointer')]")
                )
            )

            print(f"Answer options found: {len(answer_options)}")

            if answer_options:
                current_url = driver.current_url
                response = session.get(current_url)

                if response.status_code == 200:
                    raw_html = response.text

                    try:
                        cur_qa = parse_html(raw_html)
                    except ValueError:
                        print("Switching to regex extraction.")
                        cur_qa = rex_extract(raw_html)

                    if cur_qa["question"].strip().lower() not in mem:
                        has_new_question = True
                        print(f"Saved raw HTML/Json for question {question_count}.")

                        if not os.path.exists(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}"):
                            os.makedirs(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}")

                        with open(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}/{question_count}_raw.html", "w", encoding="utf-8") as file:
                            file.write(raw_html)
                        with open(f"../../log/quiz/app-emtprep-com/{level}/{section}/trial-{trial}/{question_count}.json", "w") as file:
                            json.dump(cur_qa, file, indent=4)

                # strategy 1: always select the correct answer
                if trial % 4 != 0:
                    correct_answer_text = [cur_qa["choices"][a].strip().lower() for a in cur_qa["answer"]]
                    all_options_text = [o.text.strip().lower() if o.text else o.get_attribute("innerText").strip().lower() for o in answer_options]
                    for answer_text in correct_answer_text:
                        correct_index = all_options_text.index(answer_text)
                        safe_click(driver, answer_options[correct_index])
                # strategy 2: always select the last answer
                else:
                    safe_click(driver, answer_options[-1])

                question_count += 1

                submit_button = driver.find_elements(By.XPATH, "//div[contains(text(), 'Submit')]")
                if submit_button:
                    print("✅ Submit button found. Submitting quiz.")
                    submit_button_clickable = WebDriverWait(driver, 20).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, "//div[contains(@class, 'btn-primary') and contains(text(), 'Submit') and not(contains(@class, 'btn-disabled'))]")
                        )
                    )
                    safe_click(driver, submit_button_clickable)
                    time.sleep(10)
                    break

                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//div[contains(@class, 'btn-primary') and not(contains(@class, 'btn-disabled'))]")
                    )
                )
                safe_click(driver, next_button)
                print("Next button clicked. Waiting for a new question...")
                new_question = wait_for_new_question(driver, previous_question_text)
                if not new_question:
                    print("⚠️ New question not loaded. Forcing refresh...")
                    driver.refresh()
                    time.sleep(5)

        except StaleElementReferenceException:
            print("Encountered a stale element. Refreshing...")
            driver.refresh()
            previous_question_text = ""  # Reset to avoid stale text issues
            time.sleep(5)

    driver.quit()
    if has_new_question:
        update_memory(trial, level, section)


if __name__ == "__main__":
    nums = 500
    level = "EMR"
    section = "Airway, Respiration, and Ventilation"
    for i in range(11, nums):
        print(f"Trial {i} started.")
        web_crawl(i, level, section)
        print(f"Trial {i} completed.")

        last_trial = get_last_trial_number(f"../../log/quiz/app-emtprep-com/{level}/{section}")
        if last_trial is not None and i - last_trial > 50:
            print("No new questions found in the recent 50 trials. Stopping the program.")
            break
        print("=====================================")
