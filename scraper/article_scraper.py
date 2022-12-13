from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time

class GoogleScraper:
    def __init__(self, verbose=False, log_prefix=''):
        self.__verbose = verbose
        self.__log_prefix = log_prefix
        self.__opts = Options()
        self.__opts.headless = True
        self.__driver = webdriver.Firefox(options=self.__opts)

    def __del__(self):
        self.__driver.quit()

    def __log(self, message, last=False):
        if self.__verbose:
            print(f"\r\033[K{self.__log_prefix}{message}", end="\n" if last else "")

    def find_urls_for_query(self, query):
        self.__driver.get("https://google.com")
        try:
            agree_btn = next(
                btn
                for btn in self.__driver.find_elements(
                    By.TAG_NAME, "button"
                )
                if "Ich stimme zu" in btn.get_attribute("innerHTML")
            )
            agree_btn.click()
        except:
            pass
        search_bar = self.__driver.find_element(
            By.XPATH, '//input[@title="Suche"][@name="q"]'
        )
        search_bar.send_keys(query + Keys.ENTER)

        fetch_results = lambda: [
            a_tag.get_attribute("href")
            for a_tag in self.__driver.find_elements(By.XPATH, "//a/h3/..")
        ]
        results = list()
        while not results:
            results = fetch_results()
        results_prev = list()
        while len(results_prev) < len(results):
            results_prev = results
            results = fetch_results()
            time.sleep(1)
        return results

    def find_news_urls_for_query(self, query, n_articles=10, site=None):
        self.__log(f"scraping for query {query}")
        self.__driver.get("https://news.google.com")
        self.__log(f"received news.google.com page")
        time.sleep(3)
        try:
            agree_btn = next(
                btn
                for btn in self.__driver.find_elements(
                    By.TAG_NAME, "button"
                )
                if "Accept all" in btn.get_attribute("innerHTML")
            )
            agree_btn.click()
            self.__log("agreed to google's data collection")
        except:
            pass

        time.sleep(3)
        search_bar = self.__driver.find_element(
            By.XPATH, '//input[@value="Search for topics, locations & sources"]'
        )
        action = webdriver.common.action_chains.ActionChains(
            self.__driver
        )
        action.move_to_element_with_offset(search_bar, 5, 5)
        action.click()
        if site:
            action.send_keys(f"{query} site:{site}" + Keys.ENTER)
        else:
            action.send_keys(query + Keys.ENTER)
        action.perform()
        self.__log("executed search, fetching results now")

        time.sleep(3)
        fetch_results = lambda: [
            a_tag.get_attribute("href")
            for a_tag in self.__driver.find_elements(
                By.XPATH, "//article/a"
            )
            if a_tag.is_displayed()
        ][:n_articles]

        time.sleep(10)
        results = fetch_results()
        if not results:
            self.__log("search query seems to have no results - test your search query on news.google.com", last=True)
            return list()
        results_prev = list()
        while len(results_prev) < len(results):
            results_prev = results
            results = fetch_results()
            time.sleep(3)
        self.__log("results fetched, extracting urls now")

        urls = list()
        for url in results:
            self.__driver.get(url)
            while (
                self.__driver.current_url == url
                or self.__driver.current_url == "about:blank"
            ):
                time.sleep(0.2)
            urls.append(self.__driver.current_url)
        self.__log(f"{len(urls)} news urls extracted", last=True)
        return urls

