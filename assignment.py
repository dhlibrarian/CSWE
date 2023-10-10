import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim

# Load the researchers and their interests data
researchers_data = pd.read_csv('Researchers and Interests.csv')
researchers_data['Interests'].fillna('', inplace=True)  # Fill missing values with empty strings
researchers_interests = researchers_data['Interests']
researchers_names = researchers_data['Name']

# Load the taxonomy data
taxonomy_data = pd.read_csv('topics_subtopics_abridged.csv')
taxonomy_labels = taxonomy_data['broad_topics']

# Tokenize and vectorize the researchers' interests using CountVectorizer
count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_counts = count_vectorizer.fit_transform(researchers_interests)

# Create a gensim corpus from the CountVectorizer output
corpus = gensim.matutils.Sparse2Corpus(X_counts.transpose())

# Create an LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=len(taxonomy_labels), id2word=dict((v, k) for k, v in count_vectorizer.vocabulary_.items()))

# Assign researchers to topics
researchers_by_topic = {category: [] for category in taxonomy_labels}
for idx, doc in enumerate(corpus):
    topic_weights = lda_model[doc]  # Get topic weights for each document
    top_topic = max(topic_weights, key=lambda x: x[1])[0]  # Get the topic with the highest weight
    researchers_by_topic[taxonomy_labels[top_topic]].append(researchers_names[idx])

# Create a DataFrame with taxonomy labels as rows and researchers as a comma-separated string in one column
df = pd.DataFrame({'Topic': taxonomy_labels})
df['Researchers'] = df['Topic'].apply(lambda x: ', '.join(researchers_by_topic.get(x, [])))

# Save the result to a single CSV file
df.to_csv('researchers_by_topic.csv', index=False)
