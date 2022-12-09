
import requests
import json
import pandas as pd
import os

def refresh_pricing(ticker: str="$SPX.X",
                periodType: str='day',
                period: str='250',
                frequencyType: str='daily',
                frequency: str='1',
                consumerKey: str=None
                 ) -> pd.DataFrame:
    """_summary_

    Args:
        ticker (str, optional): _description_. Defaults to ".X".
        periodType (str, optional): _description_. Defaults to 'year'.
        period (str, optional): _description_. Defaults to '20'.
        frequencyType (str, optional): _description_. Defaults to 'monthly'.
        frequency (str, optional): _description_. Defaults to '1'.

    Returns:
        pd.DataFrame: _description_
    """
    
    if not consumerKey:
        consumerKey = os.environ.get("CONSUMER_KEY")

    url = f"https://api.tdameritrade.com/v1/marketdata/{ticker}/pricehistory?periodType={periodType}&period={period}&frequencyType={frequencyType}&frequency={frequency}"
    response = requests.get(url,
            params={'apikey' : consumerKey})
    dat = pd.DataFrame(json.loads(response.content)['candles'])
    dat['date'] = pd.to_datetime(dat['datetime'],unit='ms').dt.date
    dat.index = dat['date'].values

    dat['symbol'] = ticker
    return dat[['symbol','date','close']].copy()