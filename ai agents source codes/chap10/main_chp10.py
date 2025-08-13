from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

def get_opening_hours(store: str) -> str:
    print(f"🕒 Checking opening hours for: {store}")
    hours = {
        "starbucks": "8 AM - 9 PM",
        "mcdonald's": "6 AM - 11 PM",
        "library": "10 AM - 6 PM"
    }
    return hours.get(store.lower(), "Sorry, I don't know the hours for that store.")

def call_ai(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )
    return response.output_text.strip()

def main():
    user_input = input("💬 Ask about store hours: ")
    # Step 1: Let GPT decide if it needs to use the tool
    prompt = f"""
    You are a helpful assistant that can answer questions about store opening hours.
    You also have access to this tool:
        - get_opening_hours(store: str) -> str

    If the user asks something like "When does Starbucks open?", respond ONLY with:
        get_opening_hours: store_name

    Otherwise, answer normally.

    <user-question>
    {user_input}
    </user-question>
    """
    reply = call_ai(prompt)
    print(reply)
    
    if reply.startswith("get_opening_hours:"):
        store = reply.split(":")[1].strip()
        hours = get_opening_hours(store)
        # Step 2: Final friendly response using tool result
        final_prompt = f"""
        You are a helpful assistant.
        The user asked: {user_input}
        You found the hours: {hours}

        Now respond in a friendly way using this info.
        """
        final_reply = call_ai(final_prompt)
        print("🤖", final_reply)
    else:
        print("🤖", reply) 

if __name__ == "__main__":
    main()