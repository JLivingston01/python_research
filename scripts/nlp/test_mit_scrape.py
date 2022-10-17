

import requests     
                      
from lxml import html

#Ping research page to grab MIT research themes
url = "http://www.eecs.mit.edu/research"
res = requests.get(url,headers = {'user-agent':'test useragent'})


tree = html.fromstring(res.content)

themes = tree.xpath('//li[contains(@class,"lab-theme-filter")]/span/a/text()')

themes2 = themes[-9:]

#Ping faculty go get a list of faculty, and a list of faculty with all associated information
url = "http://www.eecs.mit.edu/people/faculty-advisors"
res = requests.get(url,headers = {'user-agent':'test useragent'})


tree = html.fromstring(res.content)
people = tree.xpath('//div[contains(@class,"views-field views-field-title")]/span[contains(@class,"field-content card-title")]/a/text()')
alldata = tree.xpath('''//div[contains(@class,"view-content")]/
div[contains(@class,"people-list")]/
ul[contains(@class,"faculty-list")]/
li[contains(@class,"views-row")]/
div[contains(@class,"views-field")]/
span/a/text()''')


#If an entry in all-info is a faculty member, assign all following elements to that person's areas list, if they are research themes
#If the next element is a person, write the data to dictionary and clear areas for the next person
persondict = {}
areas = []
for i in range(len(alldata)-1):
    
    if alldata[i] in people:
        person=alldata[i]
    elif alldata[i] in themes2:
        areas.append(alldata[i])
    if alldata[i+1] in people:
        persondict[person] = areas
        areas = []
        

        




