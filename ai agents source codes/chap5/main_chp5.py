from pypdf import PdfReader
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import sqlite3

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

class Hospital(BaseModel):
    name: str = Field(..., description="name of the hospital issuing the medical report.")
    address: str = Field(..., description="address of the hospital or clinic.")
    physician: str = Field(..., description="name of the attending physician or doctor.")

class Patient(BaseModel):
    name: str = Field(..., description="The name of the patient.")
    age: int = Field(..., description="The age of the patient.")
    gender: str = Field(..., description="The gender of the patient.")
    patient_id: str = Field(..., description="The unique patient ID or record number.")

class MedicalReport(BaseModel):
    hospital: Hospital = Field(..., description="Details of the hospital issuing the report.")
    patient: Patient = Field(..., description="Details of the patient.")
    report_date: str = Field(..., description="The date the report was created.")
    diagnosis: str = Field(..., description="The diagnosis provided in the report.")
    prescription: str = Field(..., description="Prescribed medication or treatment plan.")
    notes: str = Field(..., description="Additional doctor notes or observations.")

def setup_database():
    conn = sqlite3.connect("medical_reports.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_reports (
            id INTEGER PRIMARY KEY,
            hospital_name TEXT,
            hospital_address TEXT,
            physician TEXT,
            patient_name TEXT,
            patient_age INTEGER,
            patient_gender TEXT,
            patient_id TEXT,
            report_date TEXT,
            diagnosis TEXT,
            prescription TEXT,
            notes TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_report_data(conn, report_data):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO medical_reports (
            hospital_name, hospital_address, physician,
            patient_name, patient_age, patient_gender, patient_id,
            report_date, diagnosis, prescription, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_data["hospital"]["name"],
        report_data["hospital"]["address"],
        report_data["hospital"]["physician"],
        report_data["patient"]["name"],
        report_data["patient"]["age"],
        report_data["patient"]["gender"],
        report_data["patient"]["patient_id"],
        report_data["report_date"],
        report_data["diagnosis"],
        report_data["prescription"],
        report_data["notes"]
    ))
    conn.commit()

def get_pdf_content(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        return "".join(page.extract_text() for page in reader.pages)

def extract_report_details(pdf_content: str) -> MedicalReport:
    prompt = f"""
    You are a professional medical scribe.
    Analyze the following medical report and extract structured information:
    hospital info, patient info, diagnosis, prescription, and notes.

    <medical-report>
    {pdf_content}
    </medical-report>

    Return a valid JSON object.
    """
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=prompt,
        text_format=MedicalReport,
    )
    return response.output_parsed

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py /path/to/file.pdf")
        return

    file_path = sys.argv[1]

    if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
        print("Error: Provide a valid PDF file.")
        return
    
    try:
        print(f"Processing {file_path}...")
        pdf_content = get_pdf_content(file_path)
        report = extract_report_details(pdf_content)
        conn = setup_database()
        insert_report_data(conn, report.model_dump())
        conn.close()
        print("Extracted Medical Report Details:")
        print(report.model_dump())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()