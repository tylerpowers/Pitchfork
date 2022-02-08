# Prints fine-tuned GPT-2 text generation using ReviewDataset.py
# 02-03-2022
__author__ = "tylerpowers"

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("rev_model")
tokenizer = GPT2Tokenizer.from_pretrained("rev_tokenizer")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
thing = generator("Ben Browner", max_length=200)
for item in thing:
    for val in item:
        print(item[val])
