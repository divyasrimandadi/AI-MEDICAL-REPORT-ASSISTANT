import sqlite3
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    CURRENT_DIR,
    "medical_reports.db"
)


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_tables():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        filename TEXT NOT NULL,

        prediction TEXT NOT NULL,

        confidence REAL NOT NULL,

        report TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    conn.commit()

    conn.close()


create_tables()

