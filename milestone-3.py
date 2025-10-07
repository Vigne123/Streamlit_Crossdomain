import pandas as pd
import spacy
import string

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_excel("C:\\vigne\\infosys internship 6.0\\Milestone-3\\news_category_dataset.xls")

# Filter for specific categories (uppercase)
categories_to_keep = ['SPORTS', 'ENTERTAINMENT', 'POLITICS']
df = df[df['category'].isin(categories_to_keep)]

# Keep only 'category', 'headline', 'short_description'
df = df[['category', 'headline', 'short_description']]

# Combine headline and short_description into 'text'
df['text'] = (df['headline'].fillna('') + ' ' + df['short_description'].fillna('')).str.strip()

# Make sentences short and clear (truncate to 20 words)
df['text'] = df['text'].apply(lambda x: ' '.join(x.split()[:20]))

# Preprocess: lowercase, remove punctuation, remove stopwords
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(tokens)

df['text'] = df['text'].apply(preprocess)

# Print first few rows to verify
print(df[['category', 'text']].head())
print(df.columns)

# Save the preprocessed dataset to a new Excel file
df[['category', 'text']].to_excel("C:\\vigne\\infosys internship 6.0\\Milestone-3\\preprocessed_news_category_dataset.xlsx", index=False)

triples = []

for sentence in df['text']:
    doc = nlp(sentence)
    subject = None
    relation = None
    obj = None
    # Simple extraction: subject, verb, object
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
        if token.dep_ == "ROOT":
            relation = token.text
        if token.dep_ == "dobj":
            obj = token.text
    if subject and relation and obj:
        triples.append((subject, relation, obj))

# Print triples
for triple in triples:
    print(triple)

