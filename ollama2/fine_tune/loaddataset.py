import json
from datasets import load_dataset

reddit = load_dataset('crupib/mydemo',split='train')
#reddit = reddit.to_pandas()
#reddit=reddit.to_dict(orient="records")
print(reddit)
print(reddit[0])
#with open("train.jsonl","r") as f:
#   for line in reddit:
#      print(line)
