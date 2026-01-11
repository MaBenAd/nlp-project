#!/usr/bin/env python3
"""
evaluate.py
Évalue un modèle BertForMultipleChoice sauvegardé sur le dataset RACE (dev/test).
Usage :
    python3 evaluate.py --model_dir saved_model --data_dir RACE/dev --batch_size 8
"""

import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForMultipleChoice
from sklearn.metrics import accuracy_score

from data_loader import load_race_data, RACE_Dataset


def evaluate(model, loader, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # Déplacer les tenseurs sur le device
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            # Si token_type_ids fourni par le dataset
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            labels = batch["labels"].to(device)

            # Passer labels pour obtenir la loss (optionnel)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits  # shape (batch_size, num_choices)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0

    return avg_loss, accuracy


def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    print("Loading data...")
    data = load_race_data(args.data_dir)
    print(f"Number of samples: {len(data)}")

    print("Loading tokenizer and model from", args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForMultipleChoice.from_pretrained(args.model_dir)
    model.to(device)

    dataset = RACE_Dataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    loss, acc = evaluate(model, loader, device)
    print(f"Evaluation - Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved BertForMultipleChoice model on RACE-style data."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with saved model/tokenizer (from .save_pretrained).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to RACE dev/test folder (same format que dans train.py).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional: 'cpu' or 'cuda'. If not set, auto-detect.",
    )
    args = parser.parse_args()

    main(args)
