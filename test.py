import torch
import torch.nn.functional as F
import sqlite3
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(["We are very happy", "We hope you don't hate it."])

for result in results:
    print(result)

tokens = tokenizer.tokenize("We are very happy")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy")

print(f'Tokens: {tokens}')
print(f'Token IDs: {token_ids}')
print(f'Input IDs: {input_ids}')

X_train = ["We are very happy", "We hope you don't hate it."]
batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1, 0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

save_dir = "saved"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained([save_dir])
