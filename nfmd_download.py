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
import time
#
url = "https://www.wfas.net/nfmd/public/index.php"

#Getting local session of Chrome
driver=webdriver.Chrome()
 #put here the adress of your page
driver.get(url)
###Download Fuel Moisture Data
driver.find_element_by_css_selector('#menu_form > input[type="Radio"]:nth-child(4)').click() 

dropdown_css = "#myFormD > table > tbody > tr:nth-child(1) > td:nth-child({0}) > select"
dropdowns = [dropdown_css.format(i) for i in range(1,6)]
level = dict(zip(range(1,6), dropdowns))



#def recurse(l = 1):
#    dropdown = Select(driver.find_element_by_css_selector(level[l]))
#    for selection in range(1, len(dropdown.options)):
#        dropdown.select_by_index(selection)
#        print('[LEVEL]:%s \t [SELECTION]:%s'%(l,selection))
#        if l==5:
#            driver.find_element_by_css_selector("#myFormD > table > tbody > tr:nth-child(2) > td > input").click()
#        elif l<5:                                 
#            l+=1
#            recurse(l)
#recurse()        
pause = 0.2
d1 = Select(driver.find_element_by_css_selector(level[1]))
for s1 in range(1, len(d1.options)):
    d1.select_by_index(s1)
    time.sleep(pause) 
    d2 = Select(driver.find_element_by_css_selector(level[2]))
    for s2 in range(1, len(d2.options)):
        d2.select_by_index(s2)
        time.sleep(pause) 
        d3 = Select(driver.find_element_by_css_selector(level[3]))
        for s3 in range(1, len(d3.options)):
            d3.select_by_index(s3)
            time.sleep(pause) 
            d4 = Select(driver.find_element_by_css_selector(level[4]))
            for s4 in range(1, len(d4.options)):
                d4.select_by_index(s4)
                time.sleep(pause) 
                d5 = Select(driver.find_element_by_css_selector(level[5]))
                for s5 in range(1, len(d5.options)):
                    d5.select_by_index(s5)
                    time.sleep(pause) 
                    driver.find_element_by_css_selector("#myFormD > table > tbody > tr:nth-child(2) > td > input").click()

