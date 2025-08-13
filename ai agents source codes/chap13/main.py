import json
import sqlite3
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from database import create_db_and_tables

load_dotenv()
client = OpenAI()

DB_FILE = "clinic_database.db"
create_db_and_tables()

class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
                "additionalProperties": False,
            },
        }

    def execute(self, arguments: str) -> str:
        raise NotImplementedError("Each tool must implement its own execute method.")

class VerifyPatientTool(Tool):
    def __init__(self):
        super().__init__(
            name="verify_patient",
            description="Verify patient using full name and PIN.",
            parameters={
                "name": {"type": "string", "description": "e.g., 'Sarah Lee'"},
                "pin": {"type": "string", "description": "Patient's PIN"},
            },
        )
    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            first_name, last_name = args["name"].lower().split()
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM patients WHERE LOWER(first_name) = ? AND LOWER(last_name) = ? AND pin = ?",
                (first_name, last_name, args["pin"]),
            )
            result = cursor.fetchone()
            conn.close()
            return str(result[0]) if result else str(-1)
        except Exception as e:
            return f"Error in verify_patient: {e}"
        
class GetAppointmentsTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_appointments",
            description="Get a list of a patient's upcoming appointments.",
            parameters={
                "patient_id": {"type": "integer", 
"description": "Patient ID"}
            },
        )

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM appointments WHERE patient_id = ?", (args["patient_id"],))
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return json.dumps(results)
        except Exception as e:
            return f"Error in get_appointments: {e}"

class CheckCancellationEligibilityTool(Tool):
    def __init__(self):
        super().__init__(
            name="check_cancellation_eligibility",
            description="Check if an appointment can be canceled (at least 1 day before).",
            parameters={
                "patient_id": {"type": "integer"},
                "appointment_id": {"type": "integer"},
            },
        )

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT date FROM appointments WHERE id = ? AND patient_id = ?",
                (args["appointment_id"], args["patient_id"]),
            )
            result = cursor.fetchone()
            conn.close()
            if not result:
                return str(False)
            appointment_date = datetime.fromisoformat(result[0])
            return str((appointment_date - datetime.now()).days >= 1)
        except Exception as e:
            return f"Error in check_cancellation_eligibility: {e}"

class CancelAppointmentTool(Tool):
    def __init__(self):
        super().__init__(
            name="cancel_appointment",
            description="Cancel an eligible appointment.",
            parameters={
                "patient_id": {"type": "integer"},
                "appointment_id": {"type": "integer"},
            },
        )

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            print(f"Appointment {args['appointment_id']} for patient {args['patient_id']} has been cancelled.")
            return str(True)
        except Exception as e:
            return f"Error in cancel_appointment: {e}"

class SubmitClinicFeedbackTool(Tool):
    def __init__(self):
        super().__init__(
            name="submit_clinic_feedback",
            description="Submit feedback about the clinic or visit.",
            parameters={
                "patient_id": {"type": "integer"},
                "feedback": {"type": "string"},
            },
        )

    def execute(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
            print(f"Feedback from patient {args['patient_id']}: {args['feedback']}")
            return "Thank you for your feedback!"
        except Exception as e:
            return f"Error in submit_clinic_feedback: {e}"

class ClinicAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.messages = []
        self.tools: Dict[str, Tool] = {}
        self._initialize_agent()
    
    def _register_tools(self):
        for tool in [
            VerifyPatientTool(),
            GetAppointmentsTool(),
            CheckCancellationEligibilityTool(),
            CancelAppointmentTool(),
            SubmitClinicFeedbackTool(),
        ]:
            self.tools[tool.name] = tool

    def _initialize_agent(self):
        self.messages = [
            {
                "role": "developer",
                "content": """
                You are a professional and friendly assistant at a healthcare clinic.
                Always verify the patient's identity before sharing appointment info.
                Never cancel or display appointment details for unverified patients.
                Ask for confirmation before canceling any appointment.
                Direct patients to the front desk if you're unable to help.
                """,
            }
        ]
        self._register_tools()

    def _get_tool_schemas(self) -> list[dict]:
        return [tool.get_schema() for tool in self.tools.values()]

    def run(self):
        print("Welcome to the clinic appointment assistant! How can we help you today? Type 'exit' to quit.")
        while True:
            user_input = input("Your input: ")
            if user_input.lower() == "exit":
                break

            self.messages.append({"role": "user", "content": user_input})

            for _ in range(5):
                response = self.client.responses.create(
                    model=self.model,
                    input=self.messages,
                    tools=self._get_tool_schemas(),
                )

                for reply in response.output:
                    self.messages.append(reply.model_dump())

                    if reply.type != "function_call":
                        print(reply.content[0].text)
                    else:
                        tool = self.tools.get(reply.name)
                        if tool:
                            output = tool.execute(reply.arguments)
                            self.messages.append({
                                "type": "function_call_output",
                                "call_id": reply.call_id,
                                "output": output,
                            })

                if self.messages[-1].get("type") == "message":
                    break

def main():
    agent = ClinicAgent()
    agent.run()

if __name__ == "__main__":
    main()
