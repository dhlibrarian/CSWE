from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Load the researchers' data
researchers_df = pd.read_csv('Researchers and Interests.csv')
researchers_interests = researchers_df['Interests'].tolist()

# Remove NaN values from researchers_interests
researchers_interests = [interest for interest in researchers_interests if isinstance(interest, str)]

# Custom analyzer function for vectorization
def custom_analyzer(text):
    words = text.replace(';', ',').split(', ')  # Replace semicolons with commas and then split
    phrases = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + phrases

# Create a CountVectorizer with the custom analyzer
vectorizer = CountVectorizer(analyzer=custom_analyzer, stop_words='english')
X = vectorizer.fit_transform(researchers_interests)

# Apply Latent Dirichlet Allocation (LDA)
num_topics = 50  # Adjust the number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Print the topics
def print_topics(model, vectorizer, num_top_words):
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

print_topics(lda, vectorizer, num_top_words=10)  # Adjust the number of top words per topic
