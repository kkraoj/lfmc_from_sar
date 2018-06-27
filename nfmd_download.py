# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:10:48 2018

@author: kkrao
"""

#from splinter import Browser       

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys     
from selenium.webdriver.support.select import Select 
#
url = "https://www.wfas.net/index.php/national-fuel-moisture-database-moisture-drought-103"
#with Browser() as browser: 
#  browser.visit(url)
#  browser.find_by_name('Select GACC').click()
##  copied_text = browser.find_by_id('results')[0].text


 
#Getting local session of Chrome
driver=webdriver.Chrome()
 
#put here the adress of your page
driver.get(url)

driver.switch_to_frame(driver.find_element_by_id("blockrandom"))

#button.click()

driver.find_element_by_css_selector('myformD')
driver.find_element_by_class_name('div')
driver.find_element_by_css_selector("#site_list > li:nth-child(1)").click()
driver.find_element_by_css_selector("#map > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(3) > div:nth-child(2) > div:nth-child(4) > div:nth-child(2) > div.gm-style-iw > div:nth-child(1) > div > center > table > tbody > tr:nth-child(1) > td > a").click()

#map > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(3) > div:nth-child(2) > div:nth-child(4) > div:nth-child(2) > div.gm-style-iw > div:nth-child(1) > div > center > table > tbody > tr:nth-child(1) > td > a

#https://www.wfas.net/nfmd/include/site_page.php?site=17Rd&gacc=NOCC&state=CA&grup=Lassen%20NF
#https://www.wfas.net/nfmd/include/site_page.php?site=Basin&gacc=NOCC&state=CA&grup=Lassen%20NF

ids = driver.find_element_by_name("gacc")


for ii in ids:
    #print ii.tag_name
    print(ii.get_attribute('id'))    # id name as string
    
    
    
s2= Select(driver.find_element_by_id('id_of_element'))
 
s2.select_by_value('foo')

