from importlib.metadata import version
import tiktoken


print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)  

strings = tokenizer.decode(integers)
print(strings)


# Exercise 2.1
text = "Akwirw ier"
token_ids = tokenizer.encode(text)
print("\n", token_ids)
decoded_token_ids = tokenizer.decode(token_ids)
print(decoded_token_ids)