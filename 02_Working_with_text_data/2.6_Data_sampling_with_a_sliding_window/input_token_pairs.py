import tiktoken


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print(len(enc_text)) # 5145

enc_sample = enc_text[50:] # remove first 50 tokens

# Creating input - target pairs
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x: {x}")
print(f"y:      {y}")

print("\nOriginal Text:", tokenizer.decode(enc_sample[:context_size + 1]))

print()
#      Token IDs
# Input - Target Pairs
for i in range(1, context_size + 1):
	context = enc_sample[:i]
	desired = enc_sample[i]
	print(context, "---->", desired) # left side of arrow is what LLM receives, right side of arrow is what LLM needs to predict

print()
#      Text
# Input - Target Pairs
for i in range(1, context_size + 1):
	context = enc_sample[:i]
	desired = enc_sample[i]
	print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
