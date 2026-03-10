from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    use_safetensors=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./sst2_model", safe_serialization=True)
tokenizer.save_pretrained("./sst2_model")
