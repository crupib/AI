from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def summarize_text(text: str) -> str:
    prompt = f"""
        You are a helpful assistant that summarizes text into a tweet. 
        Please summarize the following:
        <text>
        {text}
        </text>
    """

    response = client.responses.create(
        model="gpt-4o",
        input = prompt
    )
    return response.output_text

if __name__ == "__main__":
    usr_input = input("What text do you want to summarize? ")
    summary = summarize_text(usr_input)
    print("🔍 Summary:\n", summary)
