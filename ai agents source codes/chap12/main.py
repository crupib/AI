import json
import sqlite3
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from database import create_db_and_tables

load_dotenv()
client = OpenAI()

DB_FILE = "clinic_database.db"
create_db_and_tables()

def verify_patient(name: str, pin: str) -> int:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    first_name, last_name = name.lower().split()
    cursor.execute(
        "SELECT id FROM patients WHERE LOWER(first_name) = ? AND LOWER(last_name) = ? AND pin = ?",
        (first_name, last_name, pin),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else -1

def get_appointments(patient_id: int) -> List[dict]:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments WHERE patient_id = ?", (patient_id,))
    appointments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return appointments

def check_cancellation_eligibility(patient_id: int, appointment_id: int) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date FROM appointments WHERE id = ? AND patient_id = ?",
        (appointment_id, patient_id)
    )
    result = cursor.fetchone()
    conn.close()
    if not result:
        return False
    appointment_date = datetime.fromisoformat(result[0])
    return (appointment_date - datetime.now()).days >= 1

def cancel_appointment(patient_id: int, appointment_id: int) -> bool:
    print(f"Appointment {appointment_id} for patient {patient_id} has been cancelled.")
    return True

def submit_clinic_feedback(patient_id: int, feedback: str) -> str:
    print(f"Feedback from patient {patient_id}: {feedback}")
    return "Thank you for your feedback!"

available_functions = {
    "verify_patient": verify_patient,
    "get_appointments": get_appointments,
    "check_cancellation_eligibility": check_cancellation_eligibility,
    "cancel_appointment": cancel_appointment,
    "submit_clinic_feedback": submit_clinic_feedback,
}

tools = [
    {
        "type": "function",
        "name": "verify_patient",
        "description": "Verify patient using full name and PIN.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "e.g., 'Sarah Lee'"},
                "pin": {"type": "string", "description": "Patient's PIN"},
            },
            "required": ["name", "pin"]
        }
    },
    {
        "type": "function",
        "name": "get_appointments",
        "description": "Get a list of a patient's upcoming appointments.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer", "description": "Patient ID"},
            },
            "required": ["patient_id"]
        }
    },
    {
        "type": "function",
        "name": "check_cancellation_eligibility",
        "description": "Check if an appointment can be canceled (at least 1 day before).",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "appointment_id": {"type": "integer"},
            },
            "required": ["patient_id", "appointment_id"]
        }
    },
    {
        "type": "function",
        "name": "cancel_appointment",
        "description": "Cancel an eligible appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "appointment_id": {"type": "integer"},
            },
            "required": ["patient_id", "appointment_id"]
        }
    },
    {
        "type": "function",
        "name": "submit_clinic_feedback",
        "description": "Submit feedback about the clinic or visit.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "feedback": {"type": "string"},
            },
            "required": ["patient_id", "feedback"]
        }
    }
]

def execute_tool_call(tool_call) -> str:
    fn_name = tool_call.name
    fn_args = json.loads(tool_call.arguments)

    if fn_name in available_functions:
        try:
            print(f"Calling {fn_name} with arguments: {fn_args}")
            return str(available_functions[fn_name](**fn_args))
        except Exception as e:
            return f"Error calling {fn_name}: {e}"

    return f"Unknown tool: {fn_name}"

def main():
    messages = [
        {
            "role": "developer",
            "content": """
            You are a professional and friendly assistant at a healthcare clinic.
            Always verify the patient's identity before sharing appointment info.
            Never cancel or display appointment details for unverified patients.
            Ask for confirmation before canceling any appointment.
            Direct patients to the front desk if you're unable to help.
            """
        }
    ]

    print("Welcome to the clinic appointment assistant! How can we help you today? Type 'exit' to quit.")
    while True:
        user_input = input("Your input: ")
        if user_input == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        for _ in range(5):
            response = client.responses.create(
                model='gpt-4o',
                input=messages,
                tools=tools,
            )
            output = response.output

            for reply in output:
                messages.append(reply)

                if reply.type != "function_call":
                    print(reply.content[0].text)
                else:
                    tool_output = execute_tool_call(reply)
                    messages.append({
                        "type": "function_call_output",
                        "call_id": reply.call_id,
                        "output": str(tool_output),
                    })
            if not isinstance(messages[-1], dict) and messages[-1].type == "message":
                break

if __name__ == "__main__":
    main()





