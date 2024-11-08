
# Data Collection (Etherscan)

from bs4 import BeautifulSoup as bs
import requests as req
from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import math
import time
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
import numpy as np
from math import sqrt
import random

# Initialize WebDriver
driver = webdriver.Chrome()

# Navigate to the blocks page to retrieve the latest block ID
driver.get("https://etherscan.io/blocks")
time.sleep(1)
lastBlock = driver.find_element(by=By.XPATH, value="//*[@id=\"content\"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a")
lastBlockID = int(lastBlock.text)

# Generate URLs for the last 20 blocks
blockURLs = []
for blockID in range(lastBlockID, lastBlockID-20, -1):
    blockURLs.append("https://etherscan.io/txs?block=" + str(blockID))

# Function to read each row's data in the transactions table
def readRow(rowIndex):
    row = {}
    xPath = "//*[@id=\"ContentPlaceHolder1_divTransactions\"]/div[2]/table/tbody/tr[" + str(rowIndex) + "]/"
    row["txnHash"] = driver.find_element(by=By.XPATH, value=xPath + "td[2]/div/span/a").text
    row["method"] = driver.find_element(by=By.XPATH, value=xPath + "td[3]/span").text
    row["block"] = driver.find_element(by=By.XPATH, value=xPath + "td[4]/a").text
    row["age"] = driver.find_element(by=By.XPATH, value=xPath + "td[6]/span").text
    row["from"] = driver.find_element(by=By.XPATH, value=xPath + "td[8]/div/a[1]").get_attribute("href").split("/")[-1]
    row["to"] = driver.find_element(by=By.XPATH, value=xPath + "td[10]/div").find_elements(by=By.CSS_SELECTOR, value="a")[-1].get_attribute("data-clipboard-text")
    row["value"] = driver.find_element(by=By.XPATH, value=xPath + "td[11]/span").text
    row["txnFee"] = driver.find_element(by=By.XPATH, value=xPath + "td[12]").text
    return row



# Collect data for each block
table = []
for blockUrl in tqdm(blockURLs, desc="Collecting Data Blocks ("):
    driver.get(blockUrl)
    tranCount = int(driver.find_element(by=By.XPATH, value="//*[@id=\"ContentPlaceHolder1_divDataInfo\"]/div/div[1]/span").text.split(" ")[3])
    pageCount = math.ceil(tranCount / 50)
    for pageIndex in range(1, pageCount + 1):
        url = blockUrl + "&p=" + str(pageIndex)
        driver.get(url)
        time.sleep(1)
        rowBound = (tranCount - (pageCount - 1) * 50 + 1) if (pageIndex == pageCount) else 51
        for rowIndex in range(1, rowBound):
            table.append(readRow(rowIndex))
    time.sleep(1)



# Convert to DataFrame and save to CSV
dataFrame = pd.DataFrame(table)
dataFrame.to_csv("DataFrame.csv", index=False)

# # Close the driver
driver.quit()

