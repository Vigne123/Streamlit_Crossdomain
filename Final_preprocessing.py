# Preprocessing Code:

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Convert XLS to CSV
xls_file = "C:\\vigne\\infosys internship 6.0\\Milestone-3\\news_category_dataset.xls"
df = pd.read_csv(xls_file)
df.to_csv("C:\\vigne\\infosys internship 6.0\\Milestone-3\\news_category_dataset.csv", index=False)
print("Conversion done: news_category_dataset.csv created")

# Load the CSV
df = pd.read_csv("C:\\vigne\\infosys internship 6.0\\Milestone-3\\news_category_dataset.csv")

# Fill missing short_description with empty string
df['short_description'] = df['short_description'].fillna("")

# Combine headline + short_description
df['text'] = df['headline'].astype(str) + " " + df['short_description'].astype(str)

# Function to clean text
def clean_text(text):
    text = text.lower()
    # Remove links
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Keep only letters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Save preprocessed data
df.to_csv("news_dataset_cleaned.csv", index=False)
print("Preprocessing done! Saved as news_dataset_cleaned.csv")