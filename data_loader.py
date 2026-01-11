import os
import json
import torch
from torch.utils.data import Dataset


def load_race_data(path):
    data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    sample = json.load(f)
                    for i in range(len(sample["questions"])):
                        data.append(
                            {
                                "article": sample["article"],
                                "question": sample["questions"][i],
                                "options": sample["options"][i],
                                "answer": sample["answers"][i],
                            }
                        )
    return data


class RACE_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = []

        for option in item["options"]:
            encoded = self.tokenizer(
                item["article"],
                item["question"] + " " + option,
                truncation="only_first",
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            inputs.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(),
                    "attention_mask": encoded["attention_mask"].squeeze(),
                }
            )

        label = ord(item["answer"]) - ord("A")

        return {
            "input_ids": torch.stack([x["input_ids"] for x in inputs]),
            "attention_mask": torch.stack([x["attention_mask"] for x in inputs]),
            "labels": torch.tensor(label),
        }
