import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        """ Processes input text into token IDs """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """ Converts token IDs back into text """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

#           Tokenization 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Converting Tokens into Token IDs - Adding 2 new special tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print("\n Vocabular Size: ", vocab_size, "\n")

# Creating Vocabulary dictionary
vocab = {token:integer for integer, token in enumerate(all_tokens)}

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer.encode(text)
print("\n Token IDs:", ids)

decoded_ids = tokenizer.decode(ids) 
print("\n Decoded IDs:", decoded_ids)