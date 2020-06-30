import psycopg2
import pandas as pd


conn = psycopg2.connect(
    database="database",
    user="user",
    password="password",
    host="localhost",
    port="5432")

print("Opened database successfully")

cur = conn.cursor()
cur.execute('''drop table iris;''')
print("Table deleted successfully")

conn.commit()
conn.close()
