import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2",  # Make sure Llama is running locally
    "prompt": "Tell me a fun fact"
}

response = requests.post(
    url,
    json=data,
    stream=True  # Enable streaming for the response
) 


if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            # Get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)


