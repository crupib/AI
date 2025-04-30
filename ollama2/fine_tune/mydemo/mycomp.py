import tarfile
with tarfile.open('train.json2.gz','w:gz') as tar:
   tar.add('train.jsonl')
