import re


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

#           Tokenization 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("\n Sample of Tokenized output:\n", preprocessed[:30], "\n Full Token Count:", len(preprocessed))

# Converting Tokens into Token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("\n Vocabular Size:", vocab_size)

# Creating Vocabulary dictionary
vocab = {token:integer for integer, token in enumerate(all_words)}

# Printing first 51 entries of vocabulary
for i, item in enumerate(vocab.items()):
	print(item)
	if i >= 50:
		break