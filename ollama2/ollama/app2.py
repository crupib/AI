import ollama

res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "why are sky blue?"}
    ],
    stream=True
)

#print(ollama.show("llama3.2"))

for chunk in res:
	print(chunk["message"]["content"], end="", flush=True)
