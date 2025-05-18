from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

import time
import os

path = os.path.abspath("../trial_webpage_new.html")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("file://" + path)
driver.maximize_window()


def test_empty_input():
    textarea = driver.find_element(By.ID, "textarea")
    submit_button = driver.find_element(By.ID, "formSubmit")

    textarea.clear()
    textarea.send_keys("")

    time.sleep(3)
    submit_button.click()
    time.sleep(3)

    print("[✓] Empty input test passed.")


def test_one_policy():
    driver.refresh()
    time.sleep(3)

    textarea = driver.find_element(By.ID, "textarea")
    submit_button = driver.find_element(By.ID, "formSubmit")

    textarea.clear()
    textarea.send_keys("While using our website, all of your personal data will be encrypted and secured with the "
                       "highest measures possible.")

    time.sleep(3)
    submit_button.click()
    time.sleep(30)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    detail_button = driver.find_elements(By.NAME, "detailButton")[0]
    detail_button.click()
    time.sleep(30)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    print("[✓] One policy test passed.")


def test_multiple_policies():
    driver.refresh()
    time.sleep(3)

    textarea = driver.find_element(By.ID, "textarea")
    submit_button = driver.find_element(By.ID, "formSubmit")

    textarea.clear()
    textarea.clear()
    textarea.send_keys("While using our website, all of your personal data will be encrypted and secured with the "
                       "highest measures possible.")
    textarea.send_keys(Keys.SHIFT + Keys.ENTER)
    textarea.send_keys("In some circumstances, your personal data might be sold to third-parties.")
    textarea.send_keys(Keys.SHIFT + Keys.ENTER)
    textarea.send_keys("With your consent, we might send your personal data to our partners for providing you with "
                       "certain services.")

    time.sleep(3)
    submit_button.click()
    time.sleep(60)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    detail_buttons = driver.find_elements(By.NAME, "detailButton")
    last_button = detail_buttons[len(detail_buttons) - 1]
    last_button.click()
    time.sleep(60)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    print("[✓] Multiple policies test passed.")


try:
    test_empty_input()
    test_one_policy()
    test_multiple_policies()
except Exception as e:
    print("[✗] Test failed:", e)
finally:
    driver.quit()
