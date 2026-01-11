import os
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    BertTokenizer,
    BertForMultipleChoice,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score

from data_loader import load_race_data, RACE_Dataset


def main():
    # Data Importation
    print("Loading data...")
    train_data = load_race_data("RACE/train")
    dev_data = load_race_data("RACE/dev")

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of development samples: {len(dev_data)}")

    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Pytorch DataLoader
    print("Preparing datasets and dataloaders...")
    train_dataset = RACE_Dataset(train_data, tokenizer)
    dev_dataset = RACE_Dataset(dev_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=4)

    # Bert Model for Multiple Choice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    model.to(device)

    # Training Setup
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 2
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    output_dir = "saved_model"  # tu peux changer ce chemin
    os.makedirs(output_dir, exist_ok=True)

    # Training Loop
    def train_epoch(model, loader):
        model.train()
        losses = []
        all_preds = []
        all_labels = []

        for batch in tqdm(loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            # Collect predictions and labels for accuracy calculation
            predictions = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(output_dir, "training_state.pt"),
            )

        epoch_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Training Accuracy: {epoch_accuracy:.4f}")

        return np.mean(losses)

    print("Starting training...")
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader)
        print(f"Epoch {epoch + 1} | Loss : {loss:.4f}")

    # Sauvegarde du modèle & tokenizer (format Hugging Face)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
