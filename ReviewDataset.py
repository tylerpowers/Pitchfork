# Fine-tunes GPT-2 to Pitchfork reviews to create a pretentious-sounding review for any fake or real artist/album name
# I received help from Ben Browner on this project (thanks ben)
# 02-03-2022
__author__ = "tylerpowers"

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import sqlite3

path = "sqlite3.connect"


class ReviewDataset(Dataset):
    BOS = "<|beginningofsequence|>"
    EOS = "<|endofsequence|"
    PAD = "<|pad|>"

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained( # assigns beginning, end and pad tokens to each review
            "gpt2",
            bos_token=self.BOS,
            eos_token=self.EOS,
            pad_token=self.PAD,
            return_token_type_ids=False,

        )
        self.processed = []
        con = sqlite3.connect('pitchfork2.sqlite3')
        with con:
            self._process(con)
        con.close()

    def _process(self, con):
        for row in con.execute( # iterates through SQLite file
                "select body from reviews limit 100"):  # limit can be changed/removed based on your settings
            token_dict = self.tokenizer(
                self.BOS + row[0] + self.EOS,
                truncation=True,
                max_length=768,
                padding="max_length",
            )

            self.processed.append(
                (
                    torch.unsqueeze(torch.tensor(token_dict["input_ids"]), 0),
                    torch.unsqueeze(torch.tensor(token_dict["attention_mask"]), 0),
                )
            )

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]

    @property # the "train" portion of the dataset, in this case 90% of the reviews
    def train_len(self) -> int:
        return int(len(self) * 0.9)

    @property
    def eval_len(self): # the "test" portion, 10%
        return len(self) - self.train_len


print("doing stuff") # This loads the model and puts the dataset into train and test groups
dataset = ReviewDataset()
dataset.tokenizer.save_pretrained("rev_tokenizer")
config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
model.resize_token_embeddings(len(dataset.tokenizer))
train_set, eval_set = random_split(dataset, [dataset.train_len, dataset.eval_len])


def data_collector(features):
    stack = torch.stack([f[0] for f in features])

    return {
        "input_ids": stack,
        "labels": stack,
        "attention_mask": torch.stack([f[1] for f in features]),
    }


train_args = TrainingArguments( # how the model will be trained -- good to change these with your settings
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

print("doing more stuff")
trainer = Trainer( # initializes object that will actually train the data
    model=model,
    args=train_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collector
)

print("training data")
trainer.train()
trainer.save_model("./rev_model") # saves model
