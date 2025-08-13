from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
import json

tools = [
    {
        "type": "function",
        "name": "get_opening_hours",
        "description": "Get the opening hours of a specific store.",
        "parameters": {
            "type": "object",
            "properties": {
                "store": {
                    "type": "string",
                    "description": "The name of the store, e.g. Starbucks or McDonald's"
                }
            },
            "required": ["store"],
            "additionalProperties": False
        }
    }
]

def get_opening_hours(store: str) -> str:
    print(f"🕒 Checking opening hours for: {store}")
    hours = {
        "starbucks": "8 AM - 9 PM",
        "mcdonald's": "6 AM - 11 PM",
        "library": "10 AM - 6 PM"
    }
    return hours.get(store.lower(), "Sorry, I don't know the hours for that store.")

available_functions = {
    "get_opening_hours": get_opening_hours,
}

def call_ai(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )
    return response.output_text.strip()

def execute_tool_call(tool_call):
    """Execute the matched tool with given arguments."""
    fn_name = tool_call.name
    fn_args = json.loads(tool_call.arguments)

    if fn_name in available_functions:
        try:
            return available_functions[fn_name](**fn_args)
        except Exception as e:
            return f"Error calling {fn_name}: {e}"
    return f"Unknown tool: {fn_name}"

def main():
    print("🛍️ Ask me about store hours! Type 'exit' to stop.")
    
    messages = [
        {
            "role": "developer",
            "content": "You are a helpful assistant. If the user asks about store hours, use the function call when appropriate."
        }
    ]

    while True:
        user_input = input("\n💬 You: ")
        if user_input.strip().lower() == "exit":
            print("👋 Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.responses.create(
            model="gpt-4o",
            input=messages,
            tools=tools,
        )

        output = response.output[0]
        messages.append(output)

        if output.type != "function_call":
            print("🤖", response.output_text.strip())
            continue

        tool_result = execute_tool_call(output)
        messages.append({
            "type": "function_call_output",
            "call_id": output.call_id,
            "output": str(tool_result),
        })

        response = client.responses.create(
            model="gpt-4o",
            input=messages,
        )
        print("🤖", response.output_text.strip())
        messages.append(response.output[0])
        print(messages)

if __name__ == "__main__":
    main()