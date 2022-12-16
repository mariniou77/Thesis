from fastapi import FastAPI
import mysql.connector as mariadb
import connection_properties as conn
# import connection
import pandas as pd

# cursor = connection.connection()

mariadb_connection = mariadb.connect(user=conn.USER,
                                    password=conn.PASSWORD,
                                    host=conn.HOST,
                                    database=conn.DATABASE,
                                    auth_plugin='mysql_native_password')

cursor = mariadb_connection.cursor()

sql = "SELECT * FROM thesis.prices"
cursor.execute(sql)
records = cursor.fetchall()

sql1 = """SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'prices'"""
cursor.execute(sql1)
column_names = cursor.fetchall()
print(column_names)

df = pd.DataFrame(records, columns=column_names)
print(df)

app = FastAPI()

@app.get("/")
def home():
    return {"Data": cursor.rowcount}