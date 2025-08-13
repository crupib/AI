import sqlite3
from typing import List, Dict, Any

DB_FILE = "travel_database.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def create_trip_plans_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trip_plans (
        id INTEGER PRIMARY KEY,
        trip_title TEXT NOT NULL,
        trip_details TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def get_trip_plans() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trip_plans")
    trip_plans = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return trip_plans

def add_trip_plan(trip_title: str, trip_details: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trip_plans (trip_title, trip_details) VALUES (?, ?)",
        (trip_title, trip_details)
    )
    trip_plan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": trip_plan_id, "trip_title": trip_title, "trip_details": trip_details}

def delete_trip_plan(trip_plan_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM trip_plans WHERE id = ?", (trip_plan_id,))
    conn.commit()
    conn.close()

def init_db():
    create_trip_plans_table()

