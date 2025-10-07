import spacy 
import pandas as pd
from collections import Counter
import re

# 1. Load spaCy model
nlp = spacy.load("en_core_web_sm")

# 2. Helper Functions
def clean_phrase(phrase):
    """Clean a phrase: strip articles, normalize spaces, limit length."""
    if not phrase or len(phrase.strip()) < 2:
        return None
    phrase = re.sub(r'^(the|a|an)\s+', '', phrase, flags=re.IGNORECASE).strip()
    words = phrase.split()[:5]
    phrase = ' '.join(words).strip()
    if len(phrase) < 2:
        return None
    return phrase.title()

def is_meaningful_entity(token_or_chunk):
    """Check if token/chunk is a meaningful entity (proper noun, org, etc.)."""
    if hasattr(token_or_chunk, 'ents') and token_or_chunk.ents:
        return True
    if hasattr(token_or_chunk, 'pos_') and token_or_chunk.pos_ in ('PROPN', 'NOUN'):
        text_lower = token_or_chunk.text.lower()
        if text_lower not in {
            'people','country','world','time','way','thing','man','woman','year','day',
            'one','two','new','old','big','small'
        }:
            return True
    return False

def extract_triples(doc, strict_mode=True):
    """Extract clean SVO triples."""
    triples = []
    for sent in doc.sents:
        chunks = list(sent.noun_chunks)
        ents_set = {ent.text.lower() for ent in sent.ents}
        
        for token in sent:
            if token.dep_ in ("nsubj", "csubj"):
                subj_chunk = next((chunk for chunk in chunks if token in chunk), None)
                subj_tokens = [t for t in (subj_chunk or token.subtree) if t.pos_ != "PUNCT"]
                if not subj_tokens:
                    continue
                
                subject_raw = ' '.join([t.text for t in subj_tokens])
                subject = clean_phrase(subject_raw)
                if not subject or not is_meaningful_entity(token):
                    continue
                
                verb = token.head.lemma_.lower()
                skip_verbs = {"be","have","do","get","take","make","set","go","come","say","think","know","see"}
                if verb in skip_verbs:
                    continue
                
                for child in token.head.children:
                    if child.dep_ in ("dobj","pobj"):
                        obj_chunk = next((chunk for chunk in chunks if child in chunk), None)
                        obj_tokens = [t for t in (obj_chunk or child.subtree) if t.pos_ != "PUNCT"]
                        if not obj_tokens:
                            continue
                        obj_raw = ' '.join([t.text for t in obj_tokens])
                        obj = clean_phrase(obj_raw)
                        if not obj or not is_meaningful_entity(child):
                            continue
                        
                        if strict_mode and subject.lower() not in ents_set and obj.lower() not in ents_set:
                            continue
                        
                        score = 1
                        sent_ents = [e for e in sent.ents if e.text.lower() in (subject.lower(), obj.lower())]
                        if any(e.label_ in ('PERSON','ORG','GPE','MONEY','DATE') for e in sent_ents):
                            score += 2
                        if 2 <= len(subject.split()) <= 4 and 2 <= len(obj.split()) <= 4:
                            score += 1
                        
                        triples.append((subject, verb, obj, score))
    
    unique_triples = {}
    for triple in triples:
        key = (triple[0].lower(), triple[1], triple[2].lower())
        if key not in unique_triples or triple[3] > unique_triples[key][3]:
            unique_triples[key] = triple
    return list(unique_triples.values())

# 3. Load Dataset
dataset_path = "C:\\vigne\\infosys internship 6.0\\Milestone-3\\news_dataset_cleaned.csv"
try:
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} rows.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

texts = df["clean_text"].fillna("").tolist()
categories = df["category"].fillna("").tolist() if "category" in df.columns else ["" for _ in range(len(texts))]

max_texts = 10000
texts = texts[:max_texts]
categories = categories[:max_texts]
print(f"Processing {len(texts)} texts...")

# 4. Extract Triples
all_triples = []
for i, (text, cat) in enumerate(zip(texts, categories)):
    if i % 1000 == 0:
        print(f"Processed {i}/{len(texts)} texts...")
    doc = nlp(text)
    triples = extract_triples(doc, strict_mode=True)
    for t in triples:
        all_triples.append((t[0], t[1], t[2], t[3], cat))  

print(f"\nExtracted {len(all_triples)} raw triples.")

# 5. Rank by Frequency and Quality
triple_counter = Counter()
score_aggregator = {}
for subj, rel, obj, score, cat in all_triples:
    key = (subj.lower(), rel, obj.lower(), cat.lower())
    triple_counter[key] += 1
    if key not in score_aggregator:
        score_aggregator[key] = {'total_score': 0, 'count': 0}
    score_aggregator[key]['total_score'] += score
    score_aggregator[key]['count'] += 1

top_items = []
for key, freq in triple_counter.most_common(100):
    subj_lower, rel, obj_lower, cat_lower = key
    avg_score = score_aggregator[key]['total_score'] / score_aggregator[key]['count']
    top_items.append((subj_lower.title(), rel, obj_lower.title(), freq, round(avg_score,2), cat_lower.title()))

triples_df = pd.DataFrame(top_items, columns=["Entity1","Relation","Entity2","Frequency","Quality Score","Category"])

# 6. Save & Show
output_path = "meaningful_triples.csv"
triples_df.to_csv(output_path, index=False)
print(f"\n Saved top 100 triples to '{output_path}'.")
print("\nTop 20 Triples:")
print(triples_df.head(20).to_string(index=False))
