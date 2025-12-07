import re

from download_verdict import preprocessed


class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\')', r'\1', text)
def main():
    filename = "the-verdict.txt"
    tokens = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 10:
                    break

                raw_text = line.rstrip()

                preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
                preprocessed = [item.strip() for item in preprocessed if item.strip()]

                tokens.extend(preprocessed)

        print("Token count:", len(tokens))
        print(tokens)

        # Use tokens, not preprocessed
        all_words = sorted(set(tokens))
        vocab_size = len(all_words)
        print("Vocab size:", vocab_size)

        vocab = {token: integer for integer, token in enumerate(all_words)}
        for i, item in enumerate(vocab.items()):
            print(item)
            if i >= 50:
                break
        tokenizer = SimpleTokenizerV1(vocab)
        text = """"It's the last he painted, you know,"
               Mrs. GisBurn said with pardonable pride."""
        ids = tokenizer.encode(text)
        print("ids:", ids)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
