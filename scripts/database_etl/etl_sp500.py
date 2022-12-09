
from dotenv import load_dotenv
from src.database import insert_table_to_db,query_reference_stocks
from src.extract import refresh_pricing


def main():

    load_dotenv(".env",override=True)

    df = refresh_pricing(ticker='$SPX.X',
                    periodType='year',
                    period='2',
                    frequencyType='daily',
                    frequency='1'
                    )

    insert_table_to_db(df,
                    tableName='RAW_SP500',
                    if_exists='replace',
                    dbFile = "data/sqlite.db")

    return

if __name__=='__main__':
    main()
