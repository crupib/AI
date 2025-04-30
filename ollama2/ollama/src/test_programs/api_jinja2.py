from openai import OpenAI
with open('test.txt', 'r') as file:
      # Read the entire content of the file
      file_content = file.read()
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="llama3",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": file_content},
  ]
)
print(response.choices[0].message.content)
