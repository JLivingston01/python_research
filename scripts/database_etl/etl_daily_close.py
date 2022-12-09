
import pandas as pd
from dotenv import load_dotenv
from src.database import insert_table_to_db,query_reference_stocks
from src.extract import refresh_pricing
import time

def refresh_pricing_multiple(symbolDF):

    df = pd.DataFrame()
    for s in symbolDF['Symbol'].unique():
        try:
            PRC = refresh_pricing(ticker=s,
                    periodType='year',
                    period='2',
                    frequencyType='daily',
                    frequency='1'
                    )
            df=pd.concat([df,PRC])
            print(s)
            time.sleep(.4)
        except:
            pass

    return df


def main():

    symbolDF = query_reference_stocks(dbFile = "data/sqlite.db",
        targetTable = 'RAW_STOCK_UNIVERSE')

    load_dotenv(".env",override=True)

    df = refresh_pricing_multiple(symbolDF)

    insert_table_to_db(df,
                    tableName='RAW_STOCK_PRICING',
                    if_exists='replace',
                    dbFile = "data/sqlite.db")

    return

if __name__=='__main__':
    main()
