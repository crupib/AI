import re

# One canonical tokenization pattern
TOKEN_PATTERN = r'([,.:;?_!"()\']|--|\s)'


def tokenize(text: str):
    parts = re.split(TOKEN_PATTERN, text)
    return [p.strip() for p in parts if p.strip()]


class SimpleTokenizerV1:
    def __init__(self, vocab, unk_token="<UNK>"):
        # base vocab from caller
        self.str_to_int = vocab.copy()

        # add UNK if missing
        self.unk_token = unk_token
        if unk_token not in self.str_to_int:
            self.str_to_int[unk_token] = len(self.str_to_int)

        # inverse mapping
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text: str):
        tokens = tokenize(text)
        ids = [self.str_to_int.get(t, self.str_to_int[self.unk_token]) for t in tokens]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # tighten spaces before punctuation
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        # base vocab from caller
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text: str):
        preprocessed = re.split(TOKEN_PATTERN, text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # tighten spaces before punctuation
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


def main():
    filename = "the-verdict.txt"
    tokens = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                raw_text = line.rstrip()
                tokens.extend(tokenize(raw_text))

        all_tokens = sorted(list(set(tokens)))
        all_tokens.extend(["<|endoftext|>","<|unk|>"])
        vocab = {token: integer for integer, token in enumerate(all_tokens)}
        print(len(vocab.items()))

        tokenizer = SimpleTokenizerV1(vocab)


        text = "Hello, do you like tea Mr. Cellphone?"
        enc_text = tokenizer.encode(text)
        print(enc_text)
        dec_text = tokenizer.decode(enc_text)
        print(dec_text)
        text2 = "In the sunlit terraces of the palace."
        enc_text2 = tokenizer.encode(text2)
        print(enc_text2)
        dec_text2 = tokenizer.decode(enc_text2)
        print(dec_text2)
        text3="<|endoftext|>".join((text,text2))
        print(text3)
        tokenizer2 = SimpleTokenizerV2(vocab)
        print(tokenizer2.encode(text))
        print(tokenizer.decode(tokenizer2.encode(text)))
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
