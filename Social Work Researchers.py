from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import pandas as pd
nltk.download('stopwords')
import os
os.chdir(r'C:\Users\sgavel\OneDrive - University of Tennessee\Documents\Python\SW Researchers')
os.getcwd()
# Assume researchers_interests is a list of strings containing individual researchers' interests
#with open('BIPOC research areas_fulltext.txt', 'r') as file:
#    researchers_interests = file.readlines()
global cute

researchers_df = pd.read_csv('Researchers and Interests.csv')
researchers_names = researchers_df['Name'].tolist()
researchers_interests = researchers_df['Interests'].tolist()

# Remove NaN values from researchers_interests
researchers_interests = [interest for interest in researchers_interests if isinstance(interest, str)]

# Preprocess and vectorize the text
stop_words = stopwords.words('english')
#vectorizer = TfidfVectorizer(stop_words=stop_words)
#cute = vectorizer.fit_transform(researchers_interests)
# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 4))  # Include phrases up to 2 words
cute = vectorizer.fit_transform(researchers_interests)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Apply K-Means clustering
num_clusters = 100  # You can adjust this based on the desired number of clusters
#kmeans = KMeans(n_clusters=num_clusters)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
kmeans.fit(cute)

for researcher_name, cluster_label in zip(researchers_names, kmeans.labels_):
    print(f"Researcher: {researcher_name}, Cluster: {cluster_label}")

# Get cluster assignments for each researcher
researchers_clusters = dict(zip(researchers_names, kmeans.labels_))

# Print the researchers in each cluster
clusters = {}
for researcher, cluster in researchers_clusters.items():
    if cluster not in clusters:
        clusters[cluster] = [researcher]
    else:
        clusters[cluster].append(researcher)

for cluster, researchers_list in clusters.items():
    print(f"Cluster {cluster}:", researchers_list)

cluster_terms = {}
for cluster_idx, centroid in enumerate(kmeans.cluster_centers_):
    sorted_terms_idx = centroid.argsort()[::-1]  # Sort in descending order
    terms = [vectorizer.get_feature_names_out()[i] for i in sorted_terms_idx[:2]]  # Get top 10 terms
    cluster_terms[cluster_idx] = terms

# Print cluster terms
for cluster_idx, terms in cluster_terms.items():
    print(f"Cluster {cluster_idx}: {', '.join(terms)}")





