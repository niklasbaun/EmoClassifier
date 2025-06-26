import os
import time

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from Classifier.EmoDataset import EmoDataset
from Classifier.EmoClassifier import EmoClassifier

"""
make prediction for input csv file
take the text column and return the predictions
Args:
    input_csv (str): Path to the input CSV file containing text data.
"""
def predict(input_csv):
    # Load the input CSV file
    df = pd.read_csv(input_csv)

    #get text column
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    #load the model state
    model_state = torch.load('best_model_state.bin', map_location=torch.device('cpu'))
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 128

    # Create the dataset
    dataset = EmoDataset(
        texts=df['text'].values,
        #Dummy labels, as we are only predicting
        labels=np.zeros((len(df), 5)),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    # Create the DataLoader
    BATCH_SIZE = 16
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmoClassifier(n_classes=5)
    model.load_state_dict(model_state)
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()
    predictions = []
    # Iterate over the DataLoader
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get the predicted probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            # Append the predictions
            predictions.extend(probs.tolist())

    # Convert predictions to a DataFrame
    emotion_columns = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    predictions_df = pd.DataFrame(predictions, columns=emotion_columns)

    # Rename the columns to match the expected output
    predictions_df.rename(columns={col: f'predicted_{col}' for col in emotion_columns}, inplace=True)

    # Ensure the DataFrame has the correct columns
    expected_columns =[f'predicted_{col}' for col in emotion_columns]
    for col in expected_columns:
        if col not in predictions_df.columns:
            predictions_df[col] = np.nan
    predictions_df = predictions_df[expected_columns]

    #convert predicitions to binary
    predictions_df = (predictions_df > 0.5).astype(int)

    return predictions_df



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict emotions from text data in a CSV file.")
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing text data.')

    args = parser.parse_args()

    start_time = time.time()
    predictions = predict(args.input_csv)
    end_time = time.time()

    print("Predictions:")
    print(predictions.head())

    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")



