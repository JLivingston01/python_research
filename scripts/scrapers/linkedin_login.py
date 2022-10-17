import requests

from bs4 import BeautifulSoup

email = "jason.livingston@caa.columbia.edu"
password = ""

client = requests.Session()

HOMEPAGE_URL = 'https://www.linkedin.com'
LOGIN_URL = 'https://www.linkedin.com/uas/login-submit'

html = client.get(HOMEPAGE_URL).content
soup = BeautifulSoup(html, "html.parser")
csrf = soup.find('input', {'name': 'loginCsrfParam'}).get('value')

login_information = {
    'session_key': email,
    'session_password': password,
    'loginCsrfParam': csrf,
    'trk': 'guest_homepage-basic_sign-in-submit'
}

client.post(LOGIN_URL, data=login_information)

response = client.get('https://www.linkedin.com/search/results/people/?geoUrn=%5B%22103644278%22%5D&keywords=data%20scientist&origin=FACETED_SEARCH&sid=!iG')

soup = BeautifulSoup(response.text,'html.parser')
code=soup.find_all('code')

len(code)

code[9]