import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from collections import defaultdict
from EmoDataset import EmoDataset
from EmoClassifier import EmoClassifier


df = pd.read_csv('../track-a.csv')

#set labels
emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise']
df['label'] = df[emotion_columns].values.tolist()
#
#split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

#dataset
train_dataset = EmoDataset(
    texts=train_df.text.values,
    labels=train_df.label.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_dataset = EmoDataset(
    texts=val_df.text.values,
    labels=val_df.label.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

#dataloader
BATCH_SIZE = 16

train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

#model
# Initialize model
model = EmoClassifier(n_classes=len(emotion_columns))
model = model.to(device) #TODO set device (cpu/gpu)


#train
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device) #TODO set device (cpu/gpu)

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, labels)

        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += torch.sum(preds == labels.byte())

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


"""
Way to evaluate the model
"""
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, labels)

            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += torch.sum(preds == labels.byte())
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

#loop
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device, #TODO set device (cpu/gpu)
        len(train_df))

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,     #TODO set device (cpu/gpu)
        len(val_df))

    print(f'Val loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc