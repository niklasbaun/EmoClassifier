import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


"""
Load data from a CSV file and return a DataFrame.

Parameters:
file_path (str): Path to the CSV file.

Returns:
pd.DataFrame: Data loaded from the CSV file.
"""
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


"""
Preprocess the data by handling missing values and normalizing numerical features.

Parameters:
text (list of only text arguments): The only the text from the data.

Returns:
list of text (lower, stemmed).
"""
def preprocess_data(data):
    text = data['text'].tolist()
    #lowercase the text
    text = [t.lower() for t in text]

    #remove punctuation and special characters
    text = [t.replace('.', '').replace(',', '').replace('!', '').replace('?', '') for t in text]

    #remove stop words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = [' '.join([word for word in t.split() if word not in stop_words]) for t in text]
    #stem
    stemmer = PorterStemmer()
    text = [' '.join([stemmer.stem(word) for word in t.split()]) for t in text]

    #update
    data['text'] = text
    return data
