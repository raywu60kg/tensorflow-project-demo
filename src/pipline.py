import psycopg2


conn = psycopg2.connect(
    database="database",
    user="user",
    password="password", host="localhost", port="5432")
cur = conn.cursor()

cur.execute("select * from IRIS")
rows = cur.fetchall()
for row in rows:
    print(row)
conn.commit()
conn.close()
