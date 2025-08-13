from typing import Dict, Any, List
from openai import OpenAI
from tool_classes import StoreTripPlanTool, GetTripPlansTool, DeleteTripPlanTool
from dotenv import load_dotenv
load_dotenv()  # Loads the .env file
import json
from datetime import datetime

client = OpenAI()

class Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.messages: list[Dict[str, Any]] = []
        self.tools: Dict[str, Any] = {}

    def register_tool(self, tool):
        self.tools[tool.name] = tool

    def _get_tool_schemas(self) -> list[Dict[str, Any]]:
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool_call(self, tool_call: Any) -> str:
        fn_name = tool_call.name
        fn_args = json.loads(tool_call.arguments)
        if fn_name in self.tools:
            tool_to_call = self.tools[fn_name]
            try:
                return str(tool_to_call.execute(tool_call.arguments))
            except Exception as e:
                return f"Error calling {fn_name}: {e}"
        return f"Unknown tool: {fn_name}"
    
class TripPlannerAgent(Agent):
    def __init__(self):
        super().__init__()
        self.register_tool(StoreTripPlanTool())
        self.register_tool(GetTripPlansTool())
        self.register_tool(DeleteTripPlanTool())
        self._set_initial_prompt()

    def _set_initial_prompt(self):
        self.messages = [
            {
                "role": "developer",
                "content": """
                You are a helpful travel assistant. Help the user plan a custom travel itinerary.
                Ask for destination, dates, interests, travel pace, and preferences.
                Do NOT answer travel questions or give tips directly. Your job is to build the trip plan.
                """
            }
        ]

    def run(self):
        print("Hi! Let's plan your next trip.")
        while True:
            user_input = input("Your Input ('exit' to quit, 'accept' to confirm itinerary): ")
            if user_input == "exit":
                print("Exiting.")
                break
            elif user_input == "accept":
                print("Generating final itinerary plan...")
                prompt = "Create a final version of the discussed travel itinerary. Return only the plan."
                self.messages.append({"role": "user", "content": prompt})
                response = self.client.responses.create(
                    model=self.model,
                    input=self.messages,
                )
                print("\nYour Itinerary Plan:\n")
                print(response.output_text)
                return response.output_text

            self.messages.append({"role": "user", "content": user_input})
            while True:
                response = self.client.responses.create(
                    model=self.model,
                    input=self.messages,
                    tools=self._get_tool_schemas(),
                )

                reply = response.output[0]
                self.messages.append(reply)

                if reply.type != "function_call":
                    print(response.output_text)
                    break

                tool_output = self.execute_tool_call(reply)

                self.messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": reply.call_id,
                        "output": tool_output,
                    }
                )

class TravelSearchAgent(Agent):
    def __init__(self):
        super().__init__()
        self._set_initial_prompt()

    def _set_initial_prompt(self):
        self.messages = [
            {
                "role": "developer",
                "content": f"""
                You are a travel researcher. Given a trip plan, extract helpful search terms to find local attractions, events, transport options, and seasonal advice.
                Generate targeted, fresh search terms to improve trip planning.
                Today is {datetime.now().strftime('%Y-%m-%d')}.
                """
            }
        ]

    def run(self, trip_plan: str):
        from pydantic import BaseModel, Field
        from typing import List, Literal
        import os
        import requests

        class SearchConfig(BaseModel):
            search_terms: List[str]
            freshness: Literal["pd", "pw", "pm", "py"] | str = Field(...)

        print("Searching for travel information...")
        self.messages.append({
            "role": "user",
            "content": "Please generate search terms for this trip plan: " + trip_plan
        })

        response = self.client.responses.parse(
            model=self.model,
            input=self.messages,
            text_format=SearchConfig,
        )

        search = response.output_parsed
        results = []

        for search_term in search.search_terms:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": os.getenv("BRAVE_API_KEY"),
            }
            params = {
                "q": search_term,
                "count": 10,
                "freshness": search.freshness,
            }

            r = requests.get(url, headers=headers, params=params)
            result = r.json()
            if "web" in result:
                for res in result["web"].get("results", []):
                    results.append({
                        "search_term": search_term,
                        "url": res["url"],
                        "description": res["description"],
                    })
            if "news" in result:
                for res in result["news"].get("results", []):
                    results.append({
                        "search_term": search_term,
                        "url": res["url"],
                        "description": res["description"],
                    })

        return results
    
class ItineraryFormatterAgent(Agent):
    def __init__(self):
        super().__init__()
        self._set_initial_prompt()

    def _set_initial_prompt(self):
        self.messages = [
            {
                "role": "developer",
                "content": """
                You are an itinerary formatter. Format a list of search results and a travel plan into a beautiful
                Markdown document for travelers. Include URLs inline.
                Make it clean and easy to read. Return only the Markdown.
                """
            }
        ]

    def run(self, search_results: List[Dict[str, Any]], travel_plan: str):
        import json
        print("Creating an itinerary report...")
        self.messages.append({
            "role": "user",
            "content": "Integrate these search results into the travel plan in a Markdown travel itinerary: " + json.dumps(search_results, indent=2) +
            "<travelplan>" + travel_plan + "</travelplan>"
        })
        response = self.client.responses.create(
            model=self.model,
            input=self.messages,
        )
        report = response.output_text.strip()
        return report
