import ollama

with open('test.txt', 'r') as file:
    # Read the entire content of the file
    file_content = file.read()
messages = [{'role': 'user', 'content': file_content}]

for chunk in ollama.chat('llama3', messages=messages, stream=True):
      print(chunk['message']['content'], end='', flush=True)
