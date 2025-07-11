import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import defaultdict
from Classifier.EmoDataset import EmoDataset
from Classifier.EmoClassifier import EmoClassifier
import matplotlib.pyplot as plt

#load data
df = pd.read_csv('../EmoClassifier/track-a.csv')
print(torch.cuda.is_available())

# set labels
emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise']
df['label'] = df[emotion_columns].values.tolist()


# split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
MAX_LEN = 128

# training dataset
train_dataset = EmoDataset(
    texts=train_df.text.values,
    labels=train_df.label.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
#validation dataset
val_dataset = EmoDataset(
    texts=val_df.text.values,
    labels=val_df.label.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# dataloader
BATCH_SIZE = 16
#train and validation dataloaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

""" Initialize the model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmoClassifier(n_classes=len(emotion_columns))
model = model.to(device)


""" Set training parameters"""
EPOCHS = 25
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

"""
function  of what is done in each epoch
Args:
    model: The model to train
    data_loader: DataLoader for the training set
    loss_fn: Loss function used for training
    optimizer: Optimizer for updating model weights
    device: Device to run the model on (CPU or GPU)
    n_examples: Number of examples in the training set
"""


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0  # Will count FULLY correct samples

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        # Get predictions
        preds = (torch.sigmoid(outputs) > 0.5)
        correct_predictions += ((preds == labels.bool()).all(dim=1).sum().item())

        losses.append(loss.item())
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = correct_predictions / n_examples
    return accuracy, np.mean(losses)


"""
helper function to evaluate the model on the validation set
Args:
    model: The trained model to evaluate
    data_loader: DataLoader for the validation set
    loss_fn: Loss function used for evaluation
    device: Device to run the model on (CPU or GPU)
    n_examples: Number of examples in the validation set
"""
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_labels = []  # Keep as lists during collection
    all_preds = []   # Keep as lists during collection

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

            # Get binary predictions (multi-label)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Count ONLY samples where ALL labels match
            correct_predictions += ((preds == labels).all(dim=1).sum().item())
            losses.append(loss.item())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Metrics
    f1 = f1_score(all_labels, all_preds, average="macro")
    hamming = hamming_loss(all_labels, all_preds)
    accuracy = correct_predictions / n_examples
    loss = np.mean(losses)

    return accuracy, loss, f1, hamming


""" 
Main training loop
main method to train the model
used in the main.py file if no model file is found
"""

def train():
    history = defaultdict(list)
    best_accuracy = 0
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    # Train the model for a specified number of epochs
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # Train the model for one epoch
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_df))

        print(f"Training")
        print(f'Train loss {train_loss} accuracy {train_acc}')

        # Evaluate the model on the validation set
        val_acc, val_loss, val_f1, val_hamming = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_df))

        print(f"Validation")
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print(f'Val F1 {val_f1} Hamming Loss {val_hamming}')

        # Store the results in history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_hamming'].append(val_hamming)

        # Save the model if validation accuracy improves
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            # print the epoch number and validation accuracy
            print(f'Saving model at epoch {epoch + 1} with accuracy {val_acc}')
        best_accuracy = val_acc
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.show()

    #save history to a csv file
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)