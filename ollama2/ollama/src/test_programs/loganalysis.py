import ollama

with open('RTRRomfail 1.txt', 'r') as file:
    # Read the entire content of the file
    file_content = file.read()
add_text='find the errors in the log and print them out'
messages = [{'role': 'user', 'content': add_text+file_content}]

# Now file_content holds the entire content of the file
# Call the ollama.chat function with the correct messages format
for chunk in ollama.chat('mistral', messages=messages, stream=True):
    print(chunk['message']['content'], end='', flush=True)
print("\n\n\n")
