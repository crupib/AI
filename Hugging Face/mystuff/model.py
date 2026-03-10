from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
revision = 'af0f99b'

model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# Save locally with safetensors format
model.save_pretrained('./sst2_model', safe_serialization=True)
tokenizer.save_pretrained('./sst2_model')
