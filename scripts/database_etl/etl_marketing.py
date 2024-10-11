
import sqlite3
import pandas as pd
from src.database import insert_table_to_db

def main():
    data = pd.read_csv("https://huggingface.co/datasets/dianalogan/Marketing-Budget-and-Actual-Sales-Dataset/raw/main/sales_dataset.csv")

    insert_table_to_db(data,
                tableName='RAW_MARKETING_SET_1',
                if_exists='replace',
                dbFile = "data/marketing_sqlite.db")

    dataB = pd.read_csv('data/Dummy Data HSS.csv')
    
    insert_table_to_db(dataB,
                tableName='RAW_MARKETING_SET_2',
                if_exists='replace',
                dbFile = "data/marketing_sqlite.db")
                
    
    dataC = pd.read_csv('data/Advertising Budget and Sales.csv')
    
    insert_table_to_db(dataC,
                tableName='RAW_MARKETING_SET_3',
                if_exists='replace',
                dbFile = "data/marketing_sqlite.db")

    return

if __name__=='__main__':
    main()
