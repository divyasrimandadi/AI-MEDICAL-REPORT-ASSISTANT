import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "history.db"

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    prediction TEXT,
    confidence REAL,
    report TEXT
)
""")

conn.commit()


def save_prediction(
    filename,
    prediction,
    confidence,
    report
):

    cursor.execute("""
    INSERT INTO predictions(
        filename,
        prediction,
        confidence,
        report
    )
    VALUES (?, ?, ?, ?)
    """, (
        filename,
        prediction,
        confidence,
        report
    ))

    conn.commit()