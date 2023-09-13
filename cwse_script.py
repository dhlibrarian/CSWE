# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import torch
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from spellchecker.spellchecker import SpellChecker
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
from sklearn.feature_extraction.text import CountVectorizer
from symspellpy import SymSpell, Verbosity
from joblib import Parallel, delayed
import os
import pkg_resources
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(filename='script_log.txt', level=logging.INFO)

# Constants
DATA_FILE = 'interests_clean.csv'
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512

# Error Handling & Logging
def setup_environment():
    try:
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
    except Exception as e:
        logging.error(f"Error downloading necessary nltk datasets: {e}")

# Data Exploration & Visualization
def visualize_keywords(keywords):
    keywords.value_counts().head(50).plot(kind='bar', figsize=(15,7))
    plt.title('Top 50 Keywords')
    plt.ylabel('Frequency')
    plt.xlabel('Keyword')
    plt.tight_layout()
    plt.savefig('top_keywords.png')
    plt.show()

# Saving intermediates and results
def save_to_file(data, filename):
    try:
        data.to_csv(filename, index=False)
        logging.info(f"Saved data to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

# Additional Preprocessing
def remove_stopwords(text):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def get_wordnet_pos(treebank_tag):
#Map POS tag to first character used by WordNetLemmatizer
    tag = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }.get(treebank_tag[0], wordnet.NOUN)
    return tag

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens_pos = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(t, get_wordnet_pos(p)) for t, p in tokens_pos]
    clean_text = ' '.join(lemmas)
    return remove_stopwords(clean_text)

def initialize_symspell(max_edit_distance=2):
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

sym_spell = initialize_symspell()

def apply_sym_spell_correction(text):
    corrected_text = sym_spell.lookup(text, Verbosity.CLOSEST)
    if corrected_text:
        return corrected_text[0].term
    return text  # if no correction is found, return the original text

def correct_spelling_with_symspell(df: pd.DataFrame) -> pd.DataFrame:
# Corrects spelling for all entries in the DataFrame using SymSpell
    for col in df.columns:
        df[col] = df[col].apply(apply_sym_spell_correction)
    return df

def extract_embeddings(keywords: pd.Series, batch_size=32) -> np.array:
    embeddings_list = []
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size].tolist()
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        embeddings_list.extend(embeddings.numpy())
    return np.array(embeddings_list)

def load_data(filename: str) -> pd.DataFrame:
# Load CSV file as a pandas DataFrame
    return pd.read_csv(filename)

def apply_spell_check(df: pd.DataFrame) -> pd.DataFrame:
#Corrects spelling for all entries in the DataFrame
    spell = SpellChecker()
    for col in df.columns:
        df[col] = df[col].apply(spell.correction)
    return df

def get_processed_keywords(df: pd.DataFrame) -> pd.Series:
# Stack and preprocess the DataFrame to extract keywords
    return df.stack().reset_index(drop=True).str.strip()

def preprocess_keywords(keywords):
    return keywords.apply(preprocess)

def plot_frequent_keywords(keywords, top_n=25):
    keyword_freq = keywords.value_counts()
    keyword_freq.head(top_n).plot(kind='barh')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.title(f'Top {top_n} Keywords')
    plt.gca().invert_yaxis()
    plt.show()

def generate_bigrams(keywords):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    bigram_counts = vectorizer.fit_transform(keywords)
    bigrams = vectorizer.get_feature_names_out()
    bigram_freq = bigram_counts.sum(axis=0).tolist()[0]
    return dict(zip(bigrams, bigram_freq))

def plot_frequent_bigrams(bigram_dict, top_n=20):
    df_bigrams = pd.DataFrame({'bigram': list(bigram_dict.keys()), 'count': list(bigram_dict.values())})
    df_bigrams = df_bigrams.sort_values(by="count", ascending=False).head(top_n)
    plt.figure(figsize=(12, 6))
    df_bigrams.set_index('bigram').plot(kind='bar', legend=False)
    plt.title(f'Top {top_n} Bigrams')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    setup_environment()

    df = load_data(DATA_FILE)
    
    # Spell check using SymSpell
    df = correct_spelling_with_symspell(df)
    save_to_file(df, 'corrected_data_symspell.csv')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    keywords = get_processed_keywords(df)
    embeddings = extract_embeddings(keywords)
    save_to_file(pd.DataFrame(embeddings), 'embeddings.csv')
    
    # Visualize keywords
    visualize_keywords(keywords)
    
    # Visualize bigrams
    keywords_cleaned = preprocess_keywords(keywords)
    plot_frequent_keywords(keywords_cleaned, top_n=25)
    bigram_dict = generate_bigrams(keywords_cleaned)
    plot_frequent_bigrams(bigram_dict, top_n=20)