!pip install openpyxl
!python -m spacy download en_core_web_sm

import pandas as pd
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams
from nltk import FreqDist


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load data from Excel into a DataFrame
df = pd.read_excel('manual_interests.xlsx')

# Merge all terms into a single text
merged_terms = ' '.join(df['terms'])

# Process text using spaCy
doc = nlp(merged_terms)

# Extracting noun chunks (key phrases)
noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
noun_chunk_freq = Counter(noun_chunks)

# NER
entities = [ent.text for ent in doc.ents]
entity_freq = Counter(entities)

# Unigram frequency analysis
term_freq = Counter([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])

# Bi-grams and Tri-grams using NLTK
tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

bigram_freq = FreqDist(bigrams)
trigram_freq = FreqDist(trigrams)

# Top N unigrams, bigrams, and trigrams
most_common_terms = term_freq.most_common(N)
most_common_bigrams = bigram_freq.most_common(N)
most_common_trigrams = trigram_freq.most_common(N)

# Visualization: Frequency distribution of top N unigrams
plt.figure(figsize=(10, 4))
plt.bar(*zip(*most_common_terms))
plt.xticks(rotation='vertical')
plt.title('Top {} Unigrams'.format(N))
plt.show()

# Visualization: Frequency distribution of top N bigrams
plt.figure(figsize=(10, 4))
bigram_labels = [f"{bigram[0]}_{bigram[1]}" for bigram, freq in most_common_bigrams]
bigram_freqs = [freq for bigram, freq in most_common_bigrams]
plt.bar(bigram_labels, bigram_freqs)
plt.xticks(rotation='vertical')
plt.title(f'Top {N} Bigrams')
plt.show()


# Visualization: Frequency distribution of top N trigrams
plt.figure(figsize=(10, 4))
trigram_labels = [f"{trigram[0]}_{trigram[1]}_{trigram[2]}" for trigram, freq in most_common_trigrams]
trigram_freqs = [freq for trigram, freq in most_common_trigrams]
plt.bar(trigram_labels, trigram_freqs)
plt.xticks(rotation='vertical')
plt.title(f'Top {N} Trigrams')
plt.show()

# Create separate DataFrames
df_noun_chunks = pd.DataFrame({'Most_Common_NounChunks': [term for term, freq in noun_chunk_freq.most_common()]})
df_unigrams = pd.DataFrame({'Most_Common_Unigrams': [term for term, freq in most_common_terms]})
df_bigrams = pd.DataFrame({'Most_Common_Bigrams': [f"{bigram[0]}_{bigram[1]}" for bigram, freq in most_common_bigrams]})
df_trigrams = pd.DataFrame({'Most_Common_Trigrams': [f"{trigram[0]}_{trigram[1]}_{trigram[2]}" for trigram, freq in most_common_trigrams]})

# Initialize Excel writer
with pd.ExcelWriter('most_common_terms.xlsx') as writer:
    # Write each DataFrame to a specific sheet
    df_noun_chunks.to_excel(writer, sheet_name='Noun_Chunks', index=False)
    df_unigrams.to_excel(writer, sheet_name='Unigrams', index=False)
    df_bigrams.to_excel(writer, sheet_name='Bigrams', index=False)
    df_trigrams.to_excel(writer, sheet_name='Trigrams', index=False)


# You can manually categorize these terms into hierarchical groups, or use them as is
print(most_common_terms)

# Optionally: Export the list back to Excel
result_df = pd.DataFrame({'Most_Common_Terms': most_common_terms})
result_df.to_excel('most_common_terms.xlsx', index=False)
