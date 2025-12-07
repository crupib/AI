from importlib.metadata import version
import tiktoken
print("tiktoken:", tiktoken.__version__)
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

text = ("Akwirw ier")
numbers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(numbers)
strings = tokenizer.decode(numbers)
print(strings)

