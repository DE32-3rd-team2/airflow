import pymysql
import os

def get_conn():
    db_host = os.getenv("DB_IP", "52.78.181.205")
    db_port = os.getenv("DB_PORT", 23306)
    conn = pymysql.connect(
        host = db_host,
        port = int(db_port),
        user = "3rd",
        passwd = "1234",
        db = "team2",
        cursorclass=pymysql.cursors.DictCursor
        )

    return conn

def select(query: str, size = -1):
    conn = get_conn()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchmany(size)

    return result

def dml(sql, *values):
    conn = get_conn()

    with conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)
            conn.commit()
            return cursor.rowcount
