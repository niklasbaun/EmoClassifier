import csv
import time

def get_predictions(model, text):
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    probs = torch.sigmoid(output).cpu().detach().numpy()[0]

    return probs


"""
Main predict function

"""
def predict(csv):
    with open(csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            text = row[1]
            start_time = time.time()
            #TODO: get model from training
            predictions = get_predictions(model, text)
            end_time = time.time()
            print(f"Text: {text}\nPredictions: {predictions}\nTime taken: {end_time - start_time:.2f} seconds\n")

predict(csv)

