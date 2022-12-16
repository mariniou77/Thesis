import mysql.connector as mariadb
import connection_properties as conn

def connection():
    mariadb_connection = mariadb.connect(user=conn.USER,
                                        password=conn.PASSWORD,
                                        host=conn.HOST,
                                        database=conn.DATABASE,
                                        auth_plugin='mysql_native_password')
    cursor = mariadb_connection.cursor()
    return cursor

# cursor = connection()
# sql = "SELECT * FROM prices"
# cursor.execute(sql)
# records = cursor.fetchall()
# print("Total numberascascasca of rows in table: ", cursor.rowcount)