
import pandas as pd
import requests
from bs4 import BeautifulSoup


def return_soup(Url,headers):
    Page = requests.get(Url,headers=headers)
    Soup = BeautifulSoup(Page.text,features="lxml")
    return Soup

pd.set_option('display.max_columns',500)
### CONSTANTS ###
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
}

brandUrl = "https://www.laptoplist.com/brands"
brandSoup = return_soup(brandUrl,headers)

brandCards = brandSoup.find_all("div", attrs={'class':'brand-list-card'})
brandLinks = [i.find("a")['href'] for i in brandCards]


allInfo=[]
# ONE COMPANY
for companyUrl in brandLinks:
    #companyUrl = brandLinks[0]
    companySoup = return_soup(companyUrl,headers)
    
    companyTitles = companySoup.find_all("a", attrs={'class':'model-title'})
    modelLinks = [i['href'] for i in companyTitles]
    
    # ONE MODEL
    for modelUrl in modelLinks:
        print(modelUrl)
        #modelUrl = modelLinks[0]
        modelSoup=return_soup(modelUrl,headers)
        
        modelCards = modelSoup.find_all("div", attrs={'class':'laptop-list-card'})
        
        # ONE CARD
        for card in modelCards:
            
            #card = modelCards[0]
            
            name = card.find_all("h4")[0].find("a").text
            link = card.find_all("h4")[0].find("a")['href']
            brand = card.find_all("a",
                                  attrs={'class':'laptop-brand'})[0].text.replace("\n",'')
            categories = ', '.join([i.text for i in card.find_all('div',
                                attrs={'class':'laptop-list-card__category'})[0].find_all(
                                    'a'
                                    )])                 
            ram = card.find_all('span',attrs={'class':'ram'})[0].text.replace("\n","")
            ssd = card.find_all('span',attrs={'class':'ssd'})[0].text.replace("\n","")
            screen = card.find_all('span',attrs={'class':'laptop-list-screen'}
                                   )[0].text.replace("\n","")
            processor = card.find_all('span',attrs={'class':'processor'}
                                      )[0].text.replace("\n","")
            try:
                graphics_list = card.find_all('span',attrs={'class':'graphics'}
                                      )
                graphics_list = [i.text.replace("\n","") for i in graphics_list]
                graphics=' & '.join(graphics_list)
            except:
                graphics=None
            price = card.find_all('p',attrs={'class':'laptop-list-card__price'})[0].find_all(
                'span',attrs={'class':''}
                )[0].text.replace('\n','')
            
            info = [
                    name,
                    link,
                    brand,
                    categories,
                    ram,
                    ssd,
                    screen,
                    processor,
                    graphics,
                    price
                    ]
            allInfo.append(info)
            
            
infoDF = pd.DataFrame(allInfo,columns = [
    'NAME','LINK','BRAND','CATEGORIES','RAM','SSD','SCREEN','PROCESSOR',
    'GRAPHICS','PRICE'
    ])        
    
infoDF[infoDF['BRAND']=='Acer']

infoDF['PRICE'] =infoDF['PRICE'].str.replace(',','')
infoDF['PRICE'] = infoDF['PRICE'].astype(float)


