import json
from typing import Dict, Any, List
import travel_database

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
                "additionalProperties": False,
                "required": list(self.parameters.keys()),
            },
        }

    def execute(self, arguments: str) -> Any:
        raise NotImplementedError("Each tool must implement its own execute method.")
    
class StoreTripPlanTool(Tool):
    def __init__(self):
        super().__init__(
            name="store_trip_plan",
            description="Stores a user's travel plan in the database.",
            parameters={
                "trip_title": {
                    "type": "string",
                    "description": "The title of the trip (e.g., '5 Days in Tokyo')."
                },
                "trip_details": {
                    "type": "string",
                    "description": "The detailed trip plan or preferences."
                }
            },
        )

    def execute(self, arguments: str) -> Dict[str, Any]:
        args = json.loads(arguments)
        try:
            return travel_database.add_trip_plan(args["trip_title"], args["trip_details"])
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetTripPlansTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_trip_plans",
            description="Retrieves all stored travel plans.",
            parameters={},
        )

    def execute(self, arguments: str) -> List[Dict[str, Any]]:
        try:
            return travel_database.get_trip_plans()
        except Exception as e:
            return [{"status": "error", "message": str(e)}]


class DeleteTripPlanTool(Tool):
    def __init__(self):
        super().__init__(
            name="delete_trip_plan",
            description="Deletes a travel plan by ID.",
            parameters={
                "id": {
                    "type": "integer",
                    "description": "The ID of the trip plan to delete."
                }
            },
        )

    def execute(self, arguments: str) -> Dict[str, Any] | None:
        args = json.loads(arguments)
        try:
            travel_database.delete_trip_plan(args["id"])
            return {"status": "success", "deleted_id": args["id"]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

