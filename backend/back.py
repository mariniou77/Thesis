from fastapi import FastAPI
import mysql.connector as mariadb
import connection_properties as conn
import connection

# cursor = connection.connection()

mariadb_connection = mariadb.connect(user=conn.USER,
                                    password=conn.PASSWORD,
                                    host=conn.HOST,
                                    database=conn.DATABASE,
                                    auth_plugin='mysql_native_password')
                                    
cursor = mariadb_connection.cursor()

sql = "SELECT * FROM prices"
cursor.execute(sql)
records = cursor.fetchall()

app = FastAPI()

@app.get("/")
def home():
    return {"Data": cursor.rowcount}