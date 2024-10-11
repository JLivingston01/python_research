
import pandas as pd
import sqlite3 

def insert_table_to_db(df, 
    tableName, 
    if_exists='append', 
    dbFile = "data/sqlite.db"):

    conn = sqlite3.connect(dbFile)
    df.to_sql(tableName,con=conn,if_exists=if_exists,index=False)
    conn.close()

    return

def query_reference_stocks(dbFile = "data/sqlite.db",
        targetTable = 'RAW_STOCK_UNIVERSE'):

    conn = sqlite3.connect(dbFile)

    symbolDF = pd.read_sql(f"select * from {targetTable};",con=conn)

    conn.close()

    return symbolDF