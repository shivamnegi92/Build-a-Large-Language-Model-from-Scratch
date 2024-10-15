import re


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

#           Tokenization 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Converting Tokens into Token IDs - Adding 2 new special tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print("\n Vocabular Size:", vocab_size, "\n")

# Creating Vocabulary dictionary
vocab = {token:integer for integer, token in enumerate(all_tokens)}

# Printing last 5 entries of the updated vocabulary
for i, item in enumerate(list(vocab.items())[-5:]):
	print(item)