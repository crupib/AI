import sqlite3
from datetime import datetime, timedelta

def create_db_and_tables():
    conn = sqlite3.connect("clinic_database.db")
    cursor = conn.cursor()

    # Create patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        pin TEXT NOT NULL
    )
    """)

    # Create appointments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        doctor TEXT NOT NULL,
        reason TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    """)
    
    # Clear existing data (for testing/demo purposes)
    cursor.execute("DELETE FROM appointments")
    cursor.execute("DELETE FROM patients")

    # Insert sample patients
    patients = [
        ("Alice", "Tan", "1234"),
        ("Bob", "Lim", "5678")
    ]
    cursor.executemany("INSERT INTO patients (first_name, last_name, pin) VALUES (?, ?, ?)", patients)

    # Fetch generated patient IDs
    cursor.execute("SELECT id FROM patients")
    patient_ids = [row[0] for row in cursor.fetchall()]

    # Insert sample appointments
    appointments = [
        (patient_ids[0], (datetime.now() + timedelta(days=2)).isoformat(), "Dr. Lee", "Annual check-up"),
        (patient_ids[0], (datetime.now() + timedelta(days=10)).isoformat(), "Dr. Smith", "Dental cleaning"),
        (patient_ids[1], (datetime.now() + timedelta(days=1)).isoformat(), "Dr. Chan", "Follow-up visit")
    ]
    cursor.executemany("INSERT INTO appointments (patient_id, date, doctor, reason) VALUES (?, ?, ?, ?)", appointments)

    conn.commit()
    conn.close()

