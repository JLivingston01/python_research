
import pandas as pd
from src.database import insert_table_to_db

def load_source_file_excel(sourceFile='data/universe.xlsx'):

    df = pd.read_excel(sourceFile,header=1)[['Symbol','Sector','Industry']].copy()

    return df

def main():

    targetTable = 'STOCK_UNIVERSE'

    df = load_source_file_excel(sourceFile='data/universe.xlsx')

    insert_table_to_db(df,
                    tableName='RAW_STOCK_UNIVERSE',
                    if_exists='replace',
                    dbFile = "data/sqlite.db")

    return

if __name__=='__main__':
    main()
